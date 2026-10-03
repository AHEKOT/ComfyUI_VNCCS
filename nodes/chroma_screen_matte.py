"""Device-resident screen matting for illustrated characters.

Color classification, boundary unmixing and component reconstruction use Torch
operations. Only convergence flags are read on the host; pixel data stays on the
processing device until the result is returned to the caller's device.
"""

import torch
import torch.nn.functional as F


def _dot_rgb(left, right):
    # A generic reduction over just three planar channels is disproportionately
    # expensive on Metal. Explicit channel arithmetic also avoids a temporary
    # full RGB product tensor on CUDA and CPU.
    return left[:, :1] * right[:, :1] + left[:, 1:2] * right[:, 1:2] + left[:, 2:3] * right[:, 2:3]


def _box_filter(value, radius):
    kernel = radius * 2 + 1
    value = F.avg_pool2d(value, (1, kernel), 1, (0, radius), count_include_pad=False)
    return F.avg_pool2d(value, (kernel, 1), 1, (radius, 0), count_include_pad=False)


def _erode(value, radius):
    value = -F.max_pool2d(-value, (1, radius * 2 + 1), 1, (0, radius))
    return -F.max_pool2d(-value, (radius * 2 + 1, 1), 1, (radius, 0))


def _grow(marker, allowed, limit):
    for iteration in range(limit):
        previous = marker
        marker = F.max_pool2d(marker, 3, 1, 1) * allowed
        if iteration % 16 == 15 and torch.equal(previous, marker):
            break
    return marker


def _reconstruct_detail(anchor, allowed):
    """Reconstruct eight-connected detail on a sparse graph, not a dense image.

    Coarse pooling cannot decide connectivity of a one-pixel hair or gap. Only
    unresolved pixels enter this graph, which prevents coarse cells from joining
    an isolated speck to the silhouette or erasing a thin connected strand.
    """
    height, width = allowed.shape[-2:]
    seed = (anchor & allowed).flatten()
    points = torch.nonzero(allowed.flatten() & ~seed, as_tuple=False).flatten()
    count = points.numel()
    if count == 0:
        return seed.reshape_as(allowed).float()

    mapping = torch.full((height * width,), count + 1, dtype=torch.int64, device=allowed.device)
    mapping[points] = torch.arange(count, device=allowed.device)
    y, x = points // width, points % width
    neighbors = []
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        yy, xx = y + dy, x + dx
        inside = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        position = (yy * width + xx).clamp(0, height * width - 1)
        neighbor = torch.where(seed[position] & inside, count, mapping[position])
        neighbors.append(torch.where(inside, neighbor, count + 1))
    adjacency = torch.stack(neighbors, 1)
    connected = torch.zeros(count + 2, dtype=torch.bool, device=allowed.device)
    connected[count] = True
    # At most count graph edges can separate an unresolved pixel from a seed.
    for iteration in range(count):
        updated = connected[adjacency].any(1)
        converged = iteration % 16 == 15 and torch.equal(connected[:count], updated)
        connected[:count] = updated
        if converged:
            break
    output = seed.clone()
    output[points] = connected[:count]
    return output.reshape_as(allowed).float()


def _remove_plate_speckles(alpha, initial, scale, strength):
    height, width = alpha.shape[-2:]
    density = F.avg_pool2d((initial > 0.9).float(), scale, scale, ceil_mode=True)
    marker = (_box_filter(density, 1) > 0.65).float()
    support = _grow(marker, (density > 0.08).float(), 32)
    support = F.max_pool2d(support, 3, 1, 1)
    support = F.interpolate(support, size=(height, width), mode="bilinear", align_corners=False)
    # A small or entirely thin subject may have no dense anchor at all.
    supported = alpha * torch.where(marker.any(), support > 0.05, torch.ones_like(support))

    component_scale = max(1, (max(height, width) + 1023) // 1024)
    allowed = -F.max_pool2d(-(supported > 0.02).float(), component_scale, component_scale, ceil_mode=True)
    radius = max(1, round(24 * strength) // component_scale)
    anchor = _erode(allowed, radius)
    if not bool(anchor.any()):
        return alpha
    anchor = _grow(anchor, allowed, sum(allowed.shape[-2:]))
    # Repeat exact pooling cells; resizing a ceil-pooled grid shifts boundaries.
    anchor = anchor.repeat_interleave(component_scale, 2).repeat_interleave(component_scale, 3)
    anchor = anchor[:, :, :height, :width]
    # Coarse density selects seeds only. Full-resolution reconstruction must be
    # allowed to reach long, thin strands beyond the coarse support envelope.
    return alpha * _reconstruct_detail(anchor > 0.5, alpha > 0.02)


def _processing_device(image):
    if image.device.type != "cpu":
        return image.device
    # Respect ComfyUI's selected device, including an explicit CPU configuration.
    try:
        import comfy.model_management as management
    except ImportError:
        management = None
    if management is not None and hasattr(management, "get_torch_device"):
        return management.get_torch_device()
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return image.device


@torch.inference_mode()
def screen_matte(image, tolerance=0.15, softness=0.12, despill_strength=0.65,
                 edge_width=3, matte_cleanup=0.1, foreground_recover=0.35,
                 edge_decontaminate=0.75, edge_choke=0.08,
                 output_mode="straight_rgba", processing_device=None):
    """Process one HWC image and preserve its size and original output device."""
    output_device = image.device
    device = processing_device if processing_device is not None else _processing_device(image)
    source = image.to(device=device, dtype=torch.float32).clamp(0, 1)
    x = source[..., :3].permute(2, 0, 1)[None].contiguous()
    height, width = image.shape[:2]
    tolerance_scale = max(0.05, min(4.0, float(tolerance) / 0.15))
    transition = max(0.001, float(softness) * (0.8 / 0.12))

    small = F.interpolate(x, size=(64, 64), mode="nearest")
    border = torch.cat((small[:, :, :3, :].flatten(2), small[:, :, -3:, :].flatten(2),
                        small[:, :, :, :3].flatten(2), small[:, :, :, -3:].flatten(2)), dim=2)
    key = border.median(dim=2).values[:, :, None, None]
    key_delta = x - key
    rgb_distance = _dot_rgb(key_delta, key_delta).sqrt()

    # Opponent-plane angle separates hue from screen brightness and saturation.
    u, v = x[:, :1] - (x[:, 1:2] + x[:, 2:3]) * 0.5, (x[:, 1:2] - x[:, 2:3]) * 0.8660254
    ku, kv = key[:, :1] - (key[:, 1:2] + key[:, 2:3]) * 0.5, (key[:, 1:2] - key[:, 2:3]) * 0.8660254
    chroma_norm, key_norm = (u * u + v * v).sqrt(), (ku * ku + kv * kv).sqrt()
    cosine = (u * ku + v * kv) / (chroma_norm * key_norm).clamp_min(0.0001)
    hue_score = (1 - cosine) / (0.06 * tolerance_scale ** 2)
    brightness = torch.maximum(torch.maximum(x[:, :1], x[:, 1:2]), x[:, 2:3])
    # Relative saturation recognizes a dark screen/black-outline mixture while
    # excluding pale highlights. An absolute fraction of key saturation misses
    # dark green remnants and leaves an opaque green fringe around black ink.
    saturated = (chroma_norm > torch.maximum(brightness * 0.25, torch.full_like(key_norm, 0.03))) & (key_norm > 0.10)
    hue_score = torch.where(saturated, hue_score, torch.full_like(hue_score, 10.0))
    channels = key.flatten().sort().values
    pure = (channels[-2] / channels[-1].clamp_min(0.01)) < 0.15

    # With a muted plate, same-hue clothing is only removed if it connects to
    # the outer screen. Enclosed holes still use the stricter RGB key distance.
    scale = max(1, (max(height, width) + 255) // 256)
    coarse = F.avg_pool2d((hue_score < 1.5).float(), scale, scale, ceil_mode=True) > 0.85
    exterior = torch.zeros_like(coarse, dtype=torch.float32)
    exterior[:, :, 0, :] = coarse[:, :, 0, :].float()
    exterior[:, :, -1, :] = coarse[:, :, -1, :].float()
    exterior[:, :, :, 0] = coarse[:, :, :, 0].float()
    exterior[:, :, :, -1] = coarse[:, :, :, -1].float()
    exterior = _grow(exterior, coarse, sum(coarse.shape[-2:]))
    exterior = F.interpolate(exterior, size=(height, width), mode="nearest")
    # Distinct hue is positive foreground evidence on a muted plate, even if a
    # dark painted detail happens to be close to the key in RGB distance.
    muted_score = torch.maximum(rgb_distance / (0.14 * tolerance_scale),
                                torch.where(saturated, hue_score, 0.0))
    score = torch.where(pure, torch.minimum(rgb_distance / (0.16 * tolerance_scale), hue_score),
                        muted_score)
    score = torch.where((exterior > 0.5) & (hue_score < 0.65), 0.0, score)
    initial = ((score - 0.65) / transition).clamp(0, 1)

    trusted = _erode((initial > 0.98).float(), 3 if max(height, width) > 2048 else 2)
    color_scale = 2 if max(height, width) > 2048 else 1
    packed = torch.cat((x * trusted, trusted), 1)
    if color_scale > 1:
        packed = F.avg_pool2d(packed, color_scale, color_scale, ceil_mode=True)
    radius = max(1, int(edge_width) - 1)
    estimate = _box_filter(packed, radius)
    for multiplier in (2, 4):
        estimate = torch.where(estimate[:, 3:4] < 0.005, _box_filter(packed, radius * multiplier), estimate)
    if color_scale > 1:
        estimate = F.interpolate(estimate, size=(height, width), mode="bilinear", align_corners=False)
    numerator, denominator = estimate[:, :3], estimate[:, 3:4]
    foreground = numerator / denominator.clamp_min(1e-6)
    direction = foreground - key
    norm = _dot_rgb(direction, direction).clamp_min(0.001)
    alpha = (_dot_rgb(key_delta, direction) / norm).clamp(0, 1)
    residual_color = x - (key + alpha * direction)
    residual = _dot_rgb(residual_color, residual_color).sqrt()
    radius = max(0, int(edge_width))
    # Refine around confirmed screen, not around every uncertain painted pixel:
    # otherwise a dark internal hair line can make opaque neighbors transparent.
    edge = (F.max_pool2d((initial < 0.02).float(), radius * 2 + 1, 1, radius) > 0) & (denominator > 0.005) & (residual < 0.16)
    alpha = torch.where(edge, alpha, initial)
    # A black anti-aliased strand has the screen's hue. Allow color unmixing
    # immediately next to opaque foreground, while keeping flat screen clear.
    near_opaque = F.max_pool2d((initial > 0.98).float(), 3, 1, 1) > 0
    clear_screen = (score < 0.65) & (~edge | ~near_opaque | (rgb_distance < 0.025))
    alpha = torch.where(clear_screen, 0.0, alpha)
    # Black ink mixed with a pure screen has the screen's hue, but its darkness
    # gives an exact coverage estimate. Preserve these outline samples near the
    # silhouette even when nearby light hair gives a poor foreground estimate.
    near_subject = F.max_pool2d((initial > 0.98).float(), radius * 2 + 1, 1, radius) > 0
    ink_edge = pure & near_subject & (hue_score < 0.65) & (brightness < channels[-1] * 0.7)
    ink_alpha = (1 - _dot_rgb(x, key) / _dot_rgb(key, key).clamp_min(0.001)).clamp(0, 1)
    alpha = torch.where(ink_edge, ink_alpha, alpha)
    alpha_floor = max(0.001, min(0.25, float(edge_choke) * 0.25))
    alpha = torch.where(alpha < alpha_floor, 0.0, alpha)
    alpha = torch.where(alpha > 0.98, 1.0, alpha)
    if matte_cleanup > 0:
        alpha = _remove_plate_speckles(alpha, initial, scale, min(4.0, matte_cleanup / 0.1))

    # Solve C = alpha * F + (1-alpha) * B only for partial coverage.
    corrected = ((x - (1 - alpha) * key) / alpha.clamp_min(0.05)).clamp(0, 1)
    amount = min(1.0, max(0.0, despill_strength / 0.65) * max(0.0, foreground_recover / 0.35)
                 * max(0.0, edge_decontaminate / 0.75))
    rgb = torch.lerp(x, corrected, amount)
    if source.shape[-1] == 4:
        alpha = alpha * source[..., 3][None, None]
    rgb = torch.where(alpha > 0, rgb, 0.0)
    if output_mode == "premultiplied_rgba":
        rgb = rgb * alpha
    rgba = torch.cat((rgb, alpha), 1)[0].permute(1, 2, 0)
    matte = alpha[0, 0]
    debug = torch.cat((edge.float(), alpha, 1 - alpha), 1)[0].permute(1, 2, 0)
    return rgba.to(output_device), matte.to(output_device), debug.to(output_device)

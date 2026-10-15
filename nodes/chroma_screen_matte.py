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
    radius = max(1, round(8 * strength) // component_scale)
    anchor = _erode(allowed, radius)
    if not bool(anchor.any()):
        return alpha
    # Repeat exact pooling cells; resizing a ceil-pooled grid shifts boundaries.
    anchor = anchor.repeat_interleave(component_scale, 2).repeat_interleave(component_scale, 3)
    anchor = anchor[:, :, :height, :width]
    # The eroded cores are seeds only. Sparse full-resolution reconstruction
    # reaches their complete components, including long one-pixel strands.
    # Growing the dense coarse mask first produced the same seeds but launched
    # hundreds of full-frame pooling kernels.
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


def _sample_color_lines(x, background, trusted, unknown, initial, screen_cosine, remnant_band,
                        key_brightness, radius, residual_limit):
    """Select real foreground samples by color-line fit and spatial distance.

    Known-background specialization of sampling-based matting (He et al.,
    CVPR 2011). Samples are never averaged across different materials. Search
    runs only on uncertain pixels, in bounded device-resident chunks.
    """
    height, width = x.shape[-2:]
    points = torch.nonzero(unknown.flatten(), as_tuple=False).flatten()
    rgb = x[0].permute(1, 2, 0).reshape(-1, 3)
    backgrounds = background[0].permute(1, 2, 0).reshape(-1, 3)
    alpha = initial.flatten().clone()
    uncertainty = torch.zeros_like(alpha)
    chosen = rgb.clone()
    distances = sorted({max(1, round(step * max(1, radius) / 3))
                        for step in (1, 2, 3, 4, 6, 8, 12)})
    offsets = torch.tensor(
        [(dy * step, dx * step) for step in distances
         for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                        (0, 1), (1, -1), (1, 0), (1, 1))],
        device=x.device,
    )
    penalty = offsets.float().square().sum(1).sqrt() * 0.0015
    for chunk in points.split(16384):
        if chunk.numel() == 0:
            continue
        y, xpos = chunk // width, chunk % width
        sy = y[:, None] + offsets[None, :, 0]
        sx = xpos[:, None] + offsets[None, :, 1]
        valid = (sy >= 0) & (sy < height) & (sx >= 0) & (sx < width)
        indices = sy.clamp(0, height - 1) * width + sx.clamp(0, width - 1)
        valid = valid & trusted.flatten()[indices]
        candidates, colors, bg = rgb[indices], rgb[chunk], backgrounds[chunk]
        direction, delta = candidates - bg[:, None], colors - bg
        coverage = ((delta[:, None] * direction).sum(2)
                    / direction.square().sum(2).clamp_min(0.001)).clamp(0, 1)
        error = (delta[:, None] - coverage[:, :, None] * direction).square().sum(2).sqrt()
        cost = torch.where(valid, error + penalty, 1e6)
        best, which = cost.min(1)
        row = torch.arange(chunk.numel(), device=x.device)
        foreground, fitted = candidates[row, which], coverage[row, which]
        # A thin black outline may have no opaque sample at any search radius.
        # Prefer observed samples; accept this fallback only when it fits RGB.
        black = (1 - (colors * bg).sum(1) / bg.square().sum(1).clamp_min(0.001)).clamp(0, 1)
        black_error = (colors - (1 - black[:, None]) * bg).square().sum(1).sqrt()
        use_black = ((black_error + 0.025 < best) & (black_error < 0.04)
                     & (colors.amax(1) < key_brightness * 0.7))
        foreground = torch.where(use_black[:, None], 0.0, foreground)
        fitted = torch.where(use_black, black, fitted)
        residual = torch.where(use_black, black_error, error[row, which])
        # Generated plates contain color noise that is not a valid alpha mix.
        # Suppress weak, poorly explained screen remnants, not opaque outlines.
        confidence = (1 - residual / residual_limit).clamp(0, 1)
        fitted = torch.where(fitted < 0.5, fitted * confidence, fitted)
        near_screen = ((screen_cosine.flatten()[chunk] > 0.5)
                       & (delta.square().sum(1) < (key_brightness * 0.45) ** 2))
        accepted = (((best < 1e6) | use_black)
                    & ((residual < 0.08) | near_screen | remnant_band.flatten()[chunk]))
        alpha[chunk] = torch.where(accepted, fitted, alpha[chunk])
        chosen[chunk] = torch.where(accepted[:, None], foreground, colors)
        uncertainty[chunk] = ((residual / residual_limit).clamp(0, 1)
                              * accepted * (fitted > 0) * (fitted < 1))
    return (alpha.reshape(1, 1, height, width),
            chosen.reshape(height, width, 3).permute(2, 0, 1)[None],
            uncertainty.reshape(1, 1, height, width))


def _guided_alpha(alpha, guide, uncertainty, epsilon):
    """Smooth noisy fits, preserving exact color lines and rejected models."""
    mean_g, mean_a = _box_filter(guide, 1), _box_filter(alpha, 1)
    covariance = _box_filter(guide * alpha, 1) - mean_g * mean_a
    variance = (_box_filter(guide * guide, 1) - mean_g * mean_g).clamp_min(0)
    slope = covariance / (variance + epsilon)
    intercept = mean_a - slope * mean_g
    filtered = (_box_filter(slope, 1) * guide + _box_filter(intercept, 1)).clamp(0, 1)
    return torch.lerp(alpha, filtered, uncertainty)


@torch.inference_mode()
def screen_matte(image, tolerance=0.15, softness=0.12, despill_strength=0.65,
                 edge_width=3, matte_cleanup=0.1, foreground_recover=0.35,
                 edge_decontaminate=0.75, edge_choke=0.08,
                 output_mode="straight_rgba", processing_device=None):
    """Key one HWC image; retain the caller's shape, device and alpha convention.

    Screen-colored foreground is inherently ambiguous and may be lost. Do not
    fill screen-colored gaps to protect it: clean holes and accurate non-screen
    foreground take precedence. No semantic masks or model downloads are used.
    """
    output_device = image.device
    device = processing_device if processing_device is not None else _processing_device(image)
    source = image.to(device=device, dtype=torch.float32).clamp(0, 1)
    x = source[..., :3].permute(2, 0, 1)[None].contiguous()
    height, width = image.shape[:2]
    tolerance_scale = max(0.05, min(4.0, float(tolerance) / 0.15))
    softness_scale = max(0.05, min(4.0, float(softness) / 0.12))
    radius = max(0, int(edge_width))

    small = F.interpolate(x, size=(64, 64), mode="nearest")
    border = torch.cat((small[:, :, :3, :].flatten(2), small[:, :, -3:, :].flatten(2),
                        small[:, :, :, :3].flatten(2), small[:, :, :, -3:].flatten(2)), dim=2)
    key = border.median(dim=2).values[:, :, None, None]
    key_axis = torch.arange(3, device=x.device).view(1, 3, 1, 1) == key.flatten().argmax()
    primary = (x * key_axis).sum(1, keepdim=True)
    other = torch.where(key_axis, -1.0, x).amax(1, keepdim=True)
    key_other = torch.where(key_axis, -1.0, key).amax(1, keepdim=True)
    ratio = (primary - other).clamp_min(0) / (key.max() - key_other).clamp_min(0.02)
    delta = x - key
    distance = _dot_rgb(delta, delta).sqrt()
    chroma, key_chroma = x - x.mean(1, keepdim=True), key - key.mean(1, keepdim=True)
    chroma_norm, key_norm = _dot_rgb(chroma, chroma).sqrt(), _dot_rgb(key_chroma, key_chroma).sqrt()
    cosine = _dot_rgb(chroma, key_chroma) / (chroma_norm * key_norm).clamp_min(0.00001)
    plate = (((ratio > 0.8) | (cosine > 0.9)) & (distance < 0.15 * tolerance_scale))
    plate = plate | (distance < 0.035 * tolerance_scale)
    near_plate = F.max_pool2d(plate.float(), radius * 2 + 1, 1, radius) > 0.5
    near_foreground = F.max_pool2d((~plate).float(), radius * 2 + 1, 1, radius) > 0.5
    unknown = near_plate & near_foreground & (distance > 0.025 * tolerance_scale)
    # A narrow enclosed hair gap can contain a lighter/shifted screen color
    # with no exact plate pixel nearby. Include its chromatic boundary without
    # treating every similarly colored opaque interior as an unknown region.
    screen_like = (primary > other + 0.015) & (cosine > 0.5)
    thin_remnant = (screen_like & (_erode(screen_like.float(), 1) < 0.5)
                    & (distance < key.max() * 0.45) & ~plate)
    remnant_band = F.max_pool2d(thin_remnant.float(), 3, 1, 1) > 0.5
    unknown = unknown | remnant_band
    initial = (~plate).float()
    initial = torch.where(unknown & (distance < key.max() * 0.45),
                          (1 - ratio).clamp(0, 1), initial)
    trusted = ((_erode((~plate).float(), 1) > 0.5)
               & ((cosine < 0.5) | (chroma_norm < 0.025)))
    # Interior samples need not oppose the screen hue: dark blue-gray hair on
    # a blue plate still supplies valid foreground colors for its own boundary.
    trusted = trusted | ((_erode((~plate).float(), radius + 1) > 0.5)
                         & (distance > 0.25 * tolerance_scale))
    # Do not erode away the very outline whose color the edge solver needs.
    trusted = trusted | ((x.amax(1, keepdim=True) < key.max() * 0.35)
                         & ((cosine < 0.5) | (chroma_norm < 0.025)))
    # A shifted gap must not supply its own pale green pixels as foreground
    # samples. Include adjacent antialias pixels in this sampling exclusion.
    trusted = trusted & ~remnant_band
    # The trimap's opaque anchors are constraints, not pixels to re-estimate.
    # Otherwise a valid dark hair sample can itself be made translucent by a
    # different material nearby, despite already being trusted as foreground.
    unknown = unknown & ~trusted
    initial = torch.where(trusted, 1.0, initial)

    # Only background colors are averaged. A local plate estimate accommodates
    # small shading/codec shifts without borrowing colors across foreground.
    # The silhouette band already has local plate samples. Enclosed remnants
    # without such samples use the global key; wide full-frame passes are wasted.
    packed = torch.cat((x * plate, plate.float()), 1)
    estimate = _box_filter(packed, max(1, radius + 1))
    background = torch.where(estimate[:, 3:4] > 0.00001,
                             estimate[:, :3] / estimate[:, 3:4].clamp_min(0.0001), key)
    alpha, foreground, uncertainty = _sample_color_lines(
        x, background, trusted, unknown, initial, cosine, remnant_band, key.max(), radius,
        max(0.005, 0.06 * softness_scale),
    )
    alpha = _guided_alpha(alpha, distance, uncertainty, (0.01 * softness_scale) ** 2)
    alpha_floor = max(0.001, min(0.25, float(edge_choke) * 0.25))
    alpha = torch.where(alpha < alpha_floor, 0.0, alpha)
    alpha = torch.where(alpha > 0.98, 1.0, alpha)
    if matte_cleanup > 0:
        scale = max(1, (max(height, width) + 255) // 256)
        alpha = _remove_plate_speckles(alpha, initial, scale, min(4.0, matte_cleanup / 0.1))

    # Unpremultiply against the estimated plate, then regularize unstable low
    # coverage toward the selected sample. Confident opaque pixels retain RGB.
    corrected = ((x - (1 - alpha) * background) / alpha.clamp_min(0.02)).clamp(0, 1)
    corrected = torch.lerp(corrected, foreground, ((1 - alpha) * 4).clamp(0, 1))
    # Generated spill can be opaque and shift hue away from the sampled plate.
    # Desaturate its chroma, not just G/B: clamping one channel leaves cyan on
    # teal plates. Relative saturation protects nearly neutral blue-gray ink.
    corrected_chroma = corrected - corrected.mean(1, keepdim=True)
    corrected_norm = _dot_rgb(corrected_chroma, corrected_chroma).sqrt()
    corrected_cosine = (_dot_rgb(corrected_chroma, key_chroma)
                        / (corrected_norm * key_norm).clamp_min(0.00001))
    spill = ((near_plate | remnant_band) & (corrected_cosine > 0.5)
             & (corrected_norm > corrected.amax(1, keepdim=True) * 0.1))
    luma = corrected[:, :1] * 0.2126 + corrected[:, 1:2] * 0.7152 + corrected[:, 2:3] * 0.0722
    corrected = torch.where(spill, luma, corrected)
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
    debug = torch.cat((unknown.float(), alpha, 1 - alpha), 1)[0].permute(1, 2, 0)
    return rgba.to(output_device), matte.to(output_device), debug.to(output_device)


@torch.inference_mode()
def screen_matte_batch(images, tolerance=0.15, softness=0.12, despill_strength=0.65,
                       edge_width=3, matte_cleanup=0.1, foreground_recover=0.35,
                       edge_decontaminate=0.75, edge_choke=0.08,
                       output_mode="straight_rgba", processing_device=None):
    """Process one BHWC batch with a single host/device transfer each way."""
    if images.ndim != 4:
        raise ValueError("screen_matte_batch expects a BHWC image tensor")
    output_device = images.device
    device = processing_device if processing_device is not None else _processing_device(images)
    working = images.to(device=device, dtype=torch.float32).clamp(0, 1)
    results = [
        screen_matte(
            frame,
            tolerance=tolerance,
            softness=softness,
            despill_strength=despill_strength,
            edge_width=edge_width,
            matte_cleanup=matte_cleanup,
            foreground_recover=foreground_recover,
            edge_decontaminate=edge_decontaminate,
            edge_choke=edge_choke,
            output_mode=output_mode,
            processing_device=device,
        )
        for frame in working
    ]
    return tuple(torch.stack(items).to(output_device) for items in zip(*results))

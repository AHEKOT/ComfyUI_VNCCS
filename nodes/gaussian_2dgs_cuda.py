"""Fused surface rendering using the existing PyTorch CUDA release.

gsplat owns projection and tile sorting. These kernels composite RGB, normals,
ray-plane median depth and squared NDC-depth distortion in the same traversal.
All buffers belong to PyTorch and all launches use its current CUDA stream.
"""

import ctypes as C
from functools import lru_cache

import torch

from .gaussian_cuda_runtime import load_surface_kernels


_SOURCE = r'''
struct V3 { float x, y, z; };
__device__ __forceinline__ V3 cross3(V3 a, V3 b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
struct Surfel {
    float xy[2], opacity, u[3], v[3], w[3], rgb[3], normal[3], plane[3];
    int id, valid;
};
__device__ __forceinline__ Surfel load_surfel(
    int g, const float* xy, const float* transforms, const float* rgb,
    const float* opacity, const float* normals, const float* planes,
    const unsigned char* valid) {
    Surfel s;
    s.id=g; s.valid=valid[g]; s.opacity=opacity[g];
    s.xy[0]=xy[2*g]; s.xy[1]=xy[2*g+1];
    #pragma unroll
    for(int k=0;k<3;++k) {
        s.u[k]=transforms[9*g+k]; s.v[k]=transforms[9*g+3+k];
        s.w[k]=transforms[9*g+6+k]; s.rgb[k]=rgb[3*g+k];
        s.normal[k]=normals[3*g+k]; s.plane[k]=planes[3*g+k];
    }
    return s;
}
struct Hit { V3 hu,hv,c; float su,sv,dx,dy,rho3,rho2,visibility,alpha; };
__device__ __forceinline__ bool intersect(const Surfel& s,float px,float py,Hit& h) {
    h.hu={px*s.w[0]-s.u[0],px*s.w[1]-s.u[1],px*s.w[2]-s.u[2]};
    h.hv={py*s.w[0]-s.v[0],py*s.w[1]-s.v[1],py*s.w[2]-s.v[2]};
    h.c=cross3(h.hu,h.hv);
    if(h.c.z==0.f) return false;
    h.su=h.c.x/h.c.z; h.sv=h.c.y/h.c.z;
    h.dx=s.xy[0]-px; h.dy=s.xy[1]-py;
    h.rho3=h.su*h.su+h.sv*h.sv;
    h.rho2=2.f*(h.dx*h.dx+h.dy*h.dy);
    h.visibility=__expf(-0.5f*fminf(h.rho3,h.rho2));
    h.alpha=fminf(0.999f,s.opacity*h.visibility);
    return h.alpha>=1.f/255.f;
}
__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for(int offset=16;offset;offset>>=1)
        value+=__shfl_down_sync(0xffffffff,value,offset);
    return value;
}

extern "C" __global__ void surface_forward(
    const float* xy,const float* transforms,const float* rgb,const float* opacity,
    const float* normals,const float* planes,const unsigned char* valid,
    const float* reference,const float* K,const int* offsets,const int* ids,
    int width,int height,int intersections,float ndc,
    float* image,float* alphas,float* maps,float* moments,int* last,int* median) {
    __shared__ Surfel batch[256];
    const int lane=threadIdx.y*16+threadIdx.x;
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    const int tiles_x=(width+15)/16,tiles_y=(height+15)/16;
    const int tile=blockIdx.y*tiles_x+blockIdx.x;
    const int start=offsets[tile];
    const int end=tile+1<tiles_x*tiles_y?offsets[tile+1]:intersections;
    const bool inside=x<width&&y<height;
    const float px=x+0.5f,py=y+0.5f;
    const float rx=(px-K[2])/K[0],ry=(py-K[5])/K[4];
    float T=1.f,A=0.f,B=0.f,C=0.f,color[3]={},normal[3]={};
    float median_inverse=0.f;
    int last_id=start-1,median_id=-1;
    bool done=!inside;
    for(int base=start;base<end;base+=256) {
        if(__syncthreads_count(done)==256) break;
        const int index=base+lane;
        if(index<end) batch[lane]=load_surfel(ids[index],xy,transforms,rgb,opacity,normals,planes,valid);
        __syncthreads();
        const int size=min(256,end-base);
        for(int j=0;j<size&&!done;++j) {
            const Surfel& s=batch[j];
            Hit h;
            if(!intersect(s,px,py,h)) continue;
            const float next=T*(1.f-h.alpha);
            if(next<=1e-4f) { done=true; break; }
            const float weight=T*h.alpha;
            #pragma unroll
            for(int k=0;k<3;++k) {
                color[k]+=weight*s.rgb[k]; normal[k]+=weight*s.normal[k];
            }
            const float inverse=s.plane[0]*rx+s.plane[1]*ry+s.plane[2];
            if(T>0.5f) {
                median_id=s.valid&&isfinite(inverse)&&inverse>0.f&&inverse<100.f?s.id:-1;
                median_inverse=inverse;
            }
            if(s.valid) {
                const float m=ndc*inverse-reference[0];
                A+=weight; B+=weight*m; C+=weight*m*m;
            }
            T=next; last_id=base+j;
        }
        __syncthreads();
    }
    if(!inside) return;
    const int pixel=y*width+x;
    #pragma unroll
    for(int k=0;k<3;++k) { image[3*pixel+k]=color[k]; maps[6*pixel+3+k]=normal[k]; }
    alphas[pixel]=1.f-T;
    maps[6*pixel]=median_id>=0?1.f/median_inverse:0.f;
    maps[6*pixel+1]=A; maps[6*pixel+2]=fmaxf(A*C-B*B,0.f);
    moments[4*pixel]=A; moments[4*pixel+1]=B;
    moments[4*pixel+2]=C; moments[4*pixel+3]=T;
    last[pixel]=last_id; median[pixel]=median_id;
}

extern "C" __global__ void surface_backward(
    const float* xy,const float* transforms,const float* rgb,const float* opacity,
    const float* normals,const float* planes,const unsigned char* valid,
    const float* reference,const float* K,const int* offsets,const int* ids,
    int width,int height,int intersections,float ndc,
    const float* maps,const float* moments,const int* last,const int* median,
    const float* v_image,const float* v_alpha,const float* v_maps,
    float* v_xy,float* v_transform,float* v_rgb,float* v_opacity,
    float* v_normals,float* v_planes,float* v_xy_abs) {
    __shared__ Surfel batch[256];
    const int lane=threadIdx.y*16+threadIdx.x;
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    const int tiles_x=(width+15)/16,tiles_y=(height+15)/16;
    const int tile=blockIdx.y*tiles_x+blockIdx.x;
    const int start=offsets[tile];
    const int end=tile+1<tiles_x*tiles_y?offsets[tile+1]:intersections;
    const bool inside=x<width&&y<height;
    const int pixel=inside?y*width+x:0;
    const int final=inside?last[pixel]:start-1;
    int warp_final=final;
    #pragma unroll
    for(int offset=16;offset;offset>>=1)
        warp_final=max(warp_final,__shfl_xor_sync(0xffffffff,warp_final,offset));
    const float px=x+0.5f,py=y+0.5f;
    const float rx=(px-K[2])/K[0],ry=(py-K[5])/K[4];
    const float A=moments[4*pixel],B=moments[4*pixel+1],C=moments[4*pixel+2];
    const float T_final=moments[4*pixel+3];
    float T=T_final,tail_rgb[3]={},tail_normal[3]={},tail_distortion=0.f;
    float color_grad[3]={},normal_grad[3]={};
    #pragma unroll
    for(int k=0;k<3;++k) {
        color_grad[k]=inside&&v_image?v_image[3*pixel+k]:0.f;
        normal_grad[k]=inside&&v_maps?v_maps[6*pixel+3+k]:0.f;
    }
    const float ga=inside&&v_alpha?v_alpha[pixel]:0.f;
    const float gm=inside&&v_maps?v_maps[6*pixel+1]:0.f;
    const float gd=inside&&v_maps&&A*C-B*B>=0.f?v_maps[6*pixel+2]:0.f;
    const float gz=inside&&v_maps?v_maps[6*pixel]:0.f;
    for(int base=end-1;base>=start;base-=256) {
        const int index=base-lane;
        if(index>=start) batch[lane]=load_surfel(ids[index],xy,transforms,rgb,opacity,normals,planes,valid);
        __syncthreads();
        const int size=min(256,base-start+1);
        for(int j=max(0,base-warp_final);j<size;++j) {
            const Surfel& s=batch[j];
            // xy(2), transform(9), RGB(3), opacity(1), normal(3), plane(3).
            float gradient[21]={};
            if(inside&&base-j<=final) {
                Hit h;
                if(intersect(s,px,py,h)) {
                    const float reciprocal=1.f/(1.f-h.alpha);
                    T*=reciprocal;
                    const float weight=h.alpha*T;
                    float va=T_final*reciprocal*ga;
                    #pragma unroll
                    for(int k=0;k<3;++k) {
                        gradient[11+k]=weight*color_grad[k];
                        gradient[15+k]=weight*normal_grad[k];
                        va+=(s.rgb[k]*T-tail_rgb[k]*reciprocal)*color_grad[k];
                        va+=(s.normal[k]*T-tail_normal[k]*reciprocal)*normal_grad[k];
                        tail_rgb[k]+=weight*s.rgb[k]; tail_normal[k]+=weight*s.normal[k];
                    }
                    float dw=0.f,vp=0.f;
                    if(s.valid) {
                        const float inverse=s.plane[0]*rx+s.plane[1]*ry+s.plane[2];
                        const float m=ndc*inverse-reference[0];
                        dw=gm+gd*(C+A*m*m-2.f*B*m);
                        vp=2.f*weight*(A*m-B)*gd*ndc;
                    }
                    va+=T*dw-tail_distortion*reciprocal;
                    tail_distortion+=weight*dw;
                    if(s.id==median[pixel]&&gz!=0.f) {
                        const float depth=maps[6*pixel];
                        vp-=gz*depth*depth;
                    }
                    gradient[18]=vp*rx; gradient[19]=vp*ry; gradient[20]=vp;
                    if(s.opacity*h.visibility<=0.999f) {
                        const float vG=s.opacity*va;
                        gradient[14]=h.visibility*va;
                        if(h.rho3<=h.rho2) {
                            const float vu=-h.visibility*vG*h.su;
                            const float vv=-h.visibility*vG*h.sv;
                            const V3 vc={vu/h.c.z,vv/h.c.z,-(vu*h.su+vv*h.sv)/h.c.z};
                            const V3 vhu=cross3(h.hv,vc),vhv=cross3(vc,h.hu);
                            gradient[2]=-vhu.x; gradient[3]=-vhu.y; gradient[4]=-vhu.z;
                            gradient[5]=-vhv.x; gradient[6]=-vhv.y; gradient[7]=-vhv.z;
                            gradient[8]=px*vhu.x+py*vhv.x;
                            gradient[9]=px*vhu.y+py*vhv.y;
                            gradient[10]=px*vhu.z+py*vhv.z;
                        } else {
                            gradient[0]=-2.f*h.visibility*vG*h.dx;
                            gradient[1]=-2.f*h.visibility*vG*h.dy;
                        }
                    }
                }
            }
            float ax=0.f,ay=0.f;
            if(v_xy_abs) { ax=warp_sum(fabsf(gradient[0])); ay=warp_sum(fabsf(gradient[1])); }
            #pragma unroll
            for(int k=0;k<21;++k) gradient[k]=warp_sum(gradient[k]);
            if((lane&31)==0) {
                const int g=s.id;
                #pragma unroll
                for(int k=0;k<2;++k) if(gradient[k]!=0.f) atomicAdd(v_xy+2*g+k,gradient[k]);
                #pragma unroll
                for(int k=0;k<9;++k) if(gradient[2+k]!=0.f) atomicAdd(v_transform+9*g+k,gradient[2+k]);
                #pragma unroll
                for(int k=0;k<3;++k) {
                    if(gradient[11+k]!=0.f) atomicAdd(v_rgb+3*g+k,gradient[11+k]);
                    if(gradient[15+k]!=0.f) atomicAdd(v_normals+3*g+k,gradient[15+k]);
                    if(gradient[18+k]!=0.f) atomicAdd(v_planes+3*g+k,gradient[18+k]);
                }
                if(gradient[14]!=0.f) atomicAdd(v_opacity+g,gradient[14]);
                if(v_xy_abs) { if(ax!=0.f) atomicAdd(v_xy_abs+2*g,ax); if(ay!=0.f) atomicAdd(v_xy_abs+2*g+1,ay); }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ V3 world_ray(int x,int y,const float* R,const float* K) {
    const float rx=(x+0.5f-K[2])/K[0],ry=(y+0.5f-K[5])/K[4];
    return {R[0]*rx+R[3]*ry+R[6],R[1]*rx+R[4]*ry+R[7],R[2]*rx+R[5]*ry+R[8]};
}
__device__ __forceinline__ bool normal_data(
    const float* maps,const float* R,const float* K,int width,int height,int x,int y,
    V3& dx,V3& dy,V3& c,float& length) {
    if(x<1||y<1||x>=width-1||y>=height-1) return false;
    #pragma unroll
    for(int yy=-1;yy<=1;++yy) {
        #pragma unroll
        for(int xx=-1;xx<=1;++xx) {
            const int p=(y+yy)*width+x+xx;
            if(!(maps[6*p+1]>0.01f&&isfinite(maps[6*p])&&maps[6*p]>0.f)) return false;
        }
    }
    const int p=y*width+x;
    const V3 down=world_ray(x,y+1,R,K),up=world_ray(x,y-1,R,K);
    const V3 right=world_ray(x+1,y,R,K),left=world_ray(x-1,y,R,K);
    const float zd=maps[6*(p+width)],zu=maps[6*(p-width)];
    const float zr=maps[6*(p+1)],zl=maps[6*(p-1)];
    dx={zd*down.x-zu*up.x,zd*down.y-zu*up.y,zd*down.z-zu*up.z};
    dy={zr*right.x-zl*left.x,zr*right.y-zl*left.y,zr*right.z-zl*left.z};
    c=cross3(dx,dy); length=sqrtf(c.x*c.x+c.y*c.y+c.z*c.z);
    return true;
}
extern "C" __global__ void normal_forward(
    const float* maps,const float* R,const float* K,int width,int height,float* loss) {
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x>=width||y>=height) return;
    const int p=y*width+x;
    V3 dx,dy,c; float length;
    loss[p]=0.f;
    if(!normal_data(maps,R,K,width,height,x,y,dx,dy,c,length)) return;
    const float reciprocal=1.f/fmaxf(length,1e-12f);
    loss[p]=1.f-maps[6*p+1]*reciprocal*(maps[6*p+3]*c.x+maps[6*p+4]*c.y+maps[6*p+5]*c.z);
}
extern "C" __global__ void normal_backward(
    const float* maps,const float* R,const float* K,int width,int height,
    const float* v_loss,float* v_maps) {
    const int x=blockIdx.x*16+threadIdx.x,y=blockIdx.y*16+threadIdx.y;
    if(x>=width||y>=height) return;
    const int p=y*width+x;
    if(v_loss[p]==0.f) return;
    V3 dx,dy,c; float length;
    if(!normal_data(maps,R,K,width,height,x,y,dx,dy,c,length)) return;
    const float reciprocal=1.f/fmaxf(length,1e-12f);
    const float factor=-v_loss[p]*maps[6*p+1];
    const V3 n={c.x*reciprocal,c.y*reciprocal,c.z*reciprocal};
    V3 vc={factor*maps[6*p+3],factor*maps[6*p+4],factor*maps[6*p+5]};
    if(length>=1e-12f) {
        const float projection=vc.x*n.x+vc.y*n.y+vc.z*n.z;
        vc={vc.x-projection*n.x,vc.y-projection*n.y,vc.z-projection*n.z};
    }
    vc={vc.x*reciprocal,vc.y*reciprocal,vc.z*reciprocal};
    const V3 vx=cross3(dy,vc),vy=cross3(vc,dx);
    const V3 down=world_ray(x,y+1,R,K),up=world_ray(x,y-1,R,K);
    const V3 right=world_ray(x+1,y,R,K),left=world_ray(x-1,y,R,K);
    atomicAdd(v_maps+6*(p+width),vx.x*down.x+vx.y*down.y+vx.z*down.z);
    atomicAdd(v_maps+6*(p-width),-(vx.x*up.x+vx.y*up.y+vx.z*up.z));
    atomicAdd(v_maps+6*(p+1),vy.x*right.x+vy.y*right.y+vy.z*right.z);
    atomicAdd(v_maps+6*(p-1),-(vy.x*left.x+vy.y*left.y+vy.z*left.z));
    v_maps[6*p+3]=factor*n.x; v_maps[6*p+4]=factor*n.y; v_maps[6*p+5]=factor*n.z;
}
'''


@lru_cache(maxsize=None)
def _kernels(device_index):
    with torch.cuda.device(device_index):
        # A default stream handle alone may not initialize the primary context.
        _context_storage = torch.empty(1, device=torch.device("cuda", device_index))
        torch.cuda.current_stream(device_index)
        return load_surface_kernels(
            _SOURCE, ("surface_forward", "surface_backward", "normal_forward", "normal_backward"),
            torch.__file__, torch.version.cuda, torch.cuda.get_device_capability(device_index),
        )


def prepare_surface_cuda(device):
    """Initialize before the timed training loop, never at the first loss warm-up."""
    with torch.cuda.device(device):
        _kernels(device.index)


def _launch(kernel, tensors, scalars, outputs, width, height, device):
    # Pass PyTorch storage directly on its own stream; no copies or stream handoff.
    arguments = tuple(C.c_void_p(t.data_ptr() if t is not None else 0) for t in tensors)
    arguments += scalars
    arguments += tuple(C.c_void_p(t.data_ptr() if t is not None else 0) for t in outputs)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device).cuda_stream
        kernel(((width + 15) // 16, (height + 15) // 16), (16, 16), arguments, stream=stream)


class _SurfaceRasterization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, transforms, rgb, opacity, normals, planes, valid,
                reference, K, depths, densify, offsets, ids, width, height, absgrad):
        forward = _kernels(xy.device.index)[0]
        inputs = tuple(t.contiguous() for t in (xy, transforms, rgb, opacity, normals, planes, valid,
                                               reference, K, depths, densify, offsets, ids))
        xy, transforms, rgb, opacity, normals, planes, valid, reference, K, depths, densify, offsets, ids = inputs
        image = xy.new_empty((1, height, width, 3))
        alpha = xy.new_empty((1, height, width, 1))
        maps = xy.new_empty((1, height, width, 6))
        moments = xy.new_empty((height, width, 4))
        last = torch.empty((height, width), device=xy.device, dtype=torch.int32)
        median = torch.empty_like(last)
        scalars = (C.c_int(width), C.c_int(height), C.c_int(ids.numel()), C.c_float(-1_000 * 0.2 / (1_000 - 0.2)))
        _launch(forward, inputs[:9] + (offsets, ids), scalars,
                (image, alpha, maps, moments, last, median), width, height, xy.device)
        ctx.save_for_backward(*inputs, maps, moments, last, median)
        ctx.width, ctx.height, ctx.absgrad, ctx.scalars = width, height, absgrad, scalars
        ctx.set_materialize_grads(False)
        return image, alpha, maps

    @staticmethod
    def backward(ctx, v_image, v_alpha, v_maps):
        backward = _kernels(ctx.saved_tensors[0].device.index)[1]
        inputs = ctx.saved_tensors[:-4]
        maps, moments, last, median = ctx.saved_tensors[-4:]
        xy, transforms, rgb, opacity, normals, planes, valid, reference, K, depths, densify, offsets, ids = inputs
        gradients = tuple(torch.zeros_like(t) for t in inputs[:6])
        v_xy, v_transform, v_rgb, v_opacity, v_normals, v_planes = gradients
        v_abs = torch.zeros_like(xy) if ctx.absgrad else None
        upstream = tuple(t.contiguous() if t is not None else None for t in (v_image, v_alpha, v_maps))
        _launch(backward, inputs[:9] + (offsets, ids), ctx.scalars,
                (maps, moments, last, median, *upstream, *gradients, v_abs), ctx.width, ctx.height, xy.device)
        if ctx.absgrad:
            xy.absgrad = v_abs
            densify.absgrad = v_abs
        v_densify = torch.stack((v_transform[..., 0, 2], v_transform[..., 1, 2]), dim=-1) * depths[..., None]
        return (*gradients, None, None, None, None, v_densify, None, None, None, None, None)


def rasterize_surface(xy, transforms, rgb, opacity, normals, planes, valid,
                      reference, K, depths, densify, offsets, ids, width, height, absgrad=False):
    """Single-camera, unpacked float32 interface to the fused forward/backward."""
    if width < 1 or height < 1 or xy.ndim != 3 or xy.shape[0] != 1 or xy.shape[1] < 1:
        raise ValueError("Surface rasterization requires one nonempty camera and positive image dimensions")
    count = xy.shape[1]
    shapes = ((1, count, 2), (1, count, 3, 3), (1, count, 3), (1, count),
              (1, count, 3), (count, 3), (count, 1), (), (1, 3, 3), (1, count),
              (1, count, 2), (1, (height + 15) // 16, (width + 15) // 16))
    tensors = (xy, transforms, rgb, opacity, normals, planes, valid, reference, K, depths, densify, offsets)
    for index, (tensor, shape) in enumerate(zip(tensors, shapes)):
        expected_dtype = torch.bool if index == 6 else torch.int32 if index == 11 else torch.float32
        if tuple(tensor.shape) != shape or tensor.dtype != expected_dtype or tensor.device != xy.device or not tensor.is_cuda:
            raise ValueError(f"Invalid surface rasterization input {index}: expected CUDA {expected_dtype} {shape}")
    if ids.ndim != 1 or ids.dtype != torch.int32 or ids.device != xy.device:
        raise ValueError("Surface tile indices must be an int32 vector on the rendering device")
    return _SurfaceRasterization.apply(*tensors, ids, width, height, absgrad)


class _NormalConsistency(torch.autograd.Function):
    @staticmethod
    def forward(ctx, maps, rotation, K):
        maps, rotation, K = maps.contiguous(), rotation.contiguous(), K.contiguous()
        height, width = maps.shape[1:3]
        loss = maps.new_empty((1, height, width, 1))
        scalars = (C.c_int(width), C.c_int(height))
        _launch(_kernels(maps.device.index)[2], (maps, rotation, K), scalars,
                (loss,), width, height, maps.device)
        ctx.save_for_backward(maps, rotation, K)
        ctx.width, ctx.height, ctx.scalars = width, height, scalars
        return loss

    @staticmethod
    def backward(ctx, v_loss):
        maps, rotation, K = ctx.saved_tensors
        gradient = torch.zeros_like(maps)
        upstream = v_loss.contiguous()
        _launch(_kernels(maps.device.index)[3], (maps, rotation, K), ctx.scalars,
                (upstream, gradient), ctx.width, ctx.height, maps.device)
        return gradient, None, None


def normal_consistency(maps, view, intrinsics):
    """Exact central-difference normal objective, including the detached alpha mask."""
    if (not maps.is_cuda or maps.dtype != torch.float32 or maps.ndim != 4
            or maps.shape[0] != 1 or maps.shape[-1] != 6):
        raise ValueError("Normal consistency requires a single CUDA float32 six-channel geometry image")
    for tensor, shape in ((view, (1, 4, 4)), (intrinsics, (1, 3, 3))):
        if tensor.device != maps.device or tensor.dtype != maps.dtype or tuple(tensor.shape) != shape:
            raise ValueError("Normal consistency requires camera matrices on the rendering device")
    return _NormalConsistency.apply(maps, view[0, :3, :3], intrinsics).mean()

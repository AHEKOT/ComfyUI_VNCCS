"""GPU mesh projection, rasterization and texture sampling on ComfyUI's CUDA stream."""

import ctypes as C
from functools import lru_cache

import torch

from .gaussian_cuda_runtime import load_surface_kernels


_SOURCE = r'''
struct V3 { float x,y,z; };
__device__ V3 point(const float* v, int i) { return {v[3*i],v[3*i+1],v[3*i+2]}; }
__device__ V3 sub(V3 a,V3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ V3 cross(V3 a,V3 b) { return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
__device__ float dot(V3 a,V3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ V3 project(const float* m,V3 p) {
    return {m[0]*p.x+m[1]*p.y+m[2]*p.z+m[3],
            m[4]*p.x+m[5]*p.y+m[6]*p.z+m[7],
            m[8]*p.x+m[9]*p.y+m[10]*p.z+m[11]};
}
__device__ float edge(float ax,float ay,float bx,float by,float x,float y) {
    return (bx-ax)*(y-ay)-(by-ay)*(x-ax);
}
__device__ bool bary(const float* t,float x,float y,float& a,float& b,float& c) {
    float area=edge(t[0],t[1],t[2],t[3],t[4],t[5]);
    if (fabsf(area)<1.e-20f) return false;
    a=edge(t[2],t[3],t[4],t[5],x,y)/area;
    b=edge(t[4],t[5],t[0],t[1],x,y)/area;
    c=1-a-b;
    return a>=0 && b>=0 && c>=0;
}
__device__ void depth_triangle(V3 a,V3 b,V3 c,float* depth,int w,int h) {
    float t[6]={a.x/a.z,a.y/a.z,b.x/b.z,b.y/b.z,c.x/c.z,c.y/c.z};
    int x0=(int)ceilf(fminf((float)w,fmaxf(0.f,fminf(t[0],fminf(t[2],t[4]))-.5f)));
    int y0=(int)ceilf(fminf((float)h,fmaxf(0.f,fminf(t[1],fminf(t[3],t[5]))-.5f)));
    int x1=(int)floorf(fmaxf(-1.f,fminf((float)(w-1),fmaxf(t[0],fmaxf(t[2],t[4]))-.5f)));
    int y1=(int)floorf(fmaxf(-1.f,fminf((float)(h-1),fmaxf(t[1],fmaxf(t[3],t[5]))-.5f)));
    for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
        float u,v,q;
        if(bary(t,x+.5f,y+.5f,u,v,q)) {
            float z=1.f/(u/a.z+v/b.z+q/c.z);
            if(isfinite(z) && z>0) atomicMin((unsigned int*)(depth+(long long)y*w+x),__float_as_uint(z));
        }
    }
}
extern "C" __global__ void mesh_depth(const float* vertices,const int* faces,const float* matrix,
                                      float* depth,int start,int count,int w,int h) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int f=start+local;
    V3 p[5],out[5];
    for(int j=0;j<3;++j) p[j]=project(matrix,point(vertices,faces[3LL*f+j]));
    // Clip crossing triangles instead of silently losing geometry near a camera.
    int n=0;
    for(int j=0;j<3;++j) {
        V3 a=p[j],b=p[(j+1)%3]; bool ia=a.z>=1.e-6f,ib=b.z>=1.e-6f;
        if(ia) out[n++]=a;
        if(ia!=ib) {
            float t=(1.e-6f-a.z)/(b.z-a.z);
            out[n++]={a.x+t*(b.x-a.x),a.y+t*(b.y-a.y),1.e-6f};
        }
    }
    for(int j=1;j+1<n;++j) depth_triangle(out[0],out[j],out[j+1],depth,w,h);
}
extern "C" __global__ void mesh_choose_view(const float* vertices,const int* faces,const float* matrix,
    const float* center,const float* depth,const float* reference,const unsigned char* alpha,
    float* scores,int* modes,int start,int count,int w,int h,int mode,float voxel,float tolerance,float focal) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int f=start+local;
    V3 a=point(vertices,faces[3LL*f]),b=point(vertices,faces[3LL*f+1]),c=point(vertices,faces[3LL*f+2]);
    V3 p={(a.x+b.x+c.x)/3,(a.y+b.y+c.y)/3,(a.z+b.z+c.z)/3};
    V3 n=cross(sub(b,a),sub(c,a)),d={p.x-center[0],p.y-center[1],p.z-center[2]};
    float cosine=fabsf(dot(n,d))*rsqrtf(fmaxf(dot(n,n)*dot(d,d),1.e-30f));
    if(project(matrix,a).z<=1.e-6f || project(matrix,b).z<=1.e-6f || project(matrix,c).z<=1.e-6f) return;
    V3 q=project(matrix,p);
    if(q.z<=0 || cosine<.5f) return;
    int x=(int)floorf(q.x/q.z),y=(int)floorf(q.y/q.z);
    if(x<0 || x>=w || y<0 || y>=h) return;
    long long pixel=(long long)y*w+x;
    float z=depth[pixel],observed=reference[pixel];
    if(alpha[pixel]<242 || observed<=0 || !isfinite(z) ||
       fabsf(observed-q.z)>fmaxf(4*voxel,tolerance*q.z) || fabsf(z-q.z)>4*voxel) return;
    float score=cosine*cosine*cosine*cosine*focal/(q.z*q.z);
    if(score>scores[f]) { scores[f]=score; modes[f]=mode; }
}
__device__ int root(int* parent,int f) {
    int p=parent[f];
    while(p!=f) { f=p; p=parent[f]; }
    return f;
}
__device__ void join(int* parent,int a,int b) {
    while(true) {
        a=root(parent,a); b=root(parent,b);
        if(a==b) return;
        int hi=max(a,b),lo=min(a,b);
        if(atomicMin(parent+hi,lo)==hi) return;
    }
}
extern "C" __global__ void mesh_connect(const int* faces,const int* groups,unsigned long long* keys,
    int* values,int* parent,int start,int count,unsigned long long mask) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int f=start+local;
    for(int j=0;j<3;++j) {
        unsigned long long key=((unsigned long long)(unsigned int)faces[3LL*f+j]<<32)|(unsigned int)groups[f];
        unsigned long long hash=key;
        hash^=hash>>33; hash*=0xff51afd7ed558ccdULL; hash^=hash>>33;
        unsigned long long slot=hash&mask;
        while(true) {
            unsigned long long previous=atomicCAS(keys+slot,~0ULL,key);
            if(previous==~0ULL || previous==key) {
                int other=atomicCAS(values+slot,-1,f);
                if(other>=0) join(parent,f,other);
                break;
            }
            slot=(slot+1)&mask;
        }
    }
}
extern "C" __global__ void mesh_compress(int* parent,int start,int count) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local<count) { int f=start+local; parent[f]=root(parent,f); }
}
extern "C" __global__ void mesh_uv_raster(const float* uv,int* owner,
    int start,int count,int w,int h) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int f=start+local; const float* t=uv+6LL*f;
    float p[6]; for(int j=0;j<6;++j) p[j]=t[j]*(j%2?h:w);
    int x0=max(0,(int)ceilf(fminf(p[0],fminf(p[2],p[4]))-.5f));
    int y0=max(0,(int)ceilf(fminf(p[1],fminf(p[3],p[5]))-.5f));
    int x1=min(w-1,(int)floorf(fmaxf(p[0],fmaxf(p[2],p[4]))-.5f));
    int y1=min(h-1,(int)floorf(fmaxf(p[1],fmaxf(p[3],p[5]))-.5f));
    for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
        float a,b,c;
        if(!bary(p,x+.5f,y+.5f,a,b,c)) continue;
        long long pixel=(long long)y*w+x;
        atomicMin(owner+pixel,f);
    }
}
__device__ void uv_bounds(const float* t,int w,int h,int& x0,int& y0,int& x1,int& y1) {
    x0=max(0,min(w-1,(int)floorf(fminf(t[0],fminf(t[2],t[4]))*w)));
    y0=max(0,min(h-1,(int)floorf(fminf(t[1],fminf(t[3],t[5]))*h)));
    x1=max(0,min(w-1,(int)floorf(fmaxf(t[0],fmaxf(t[2],t[4]))*w)));
    y1=max(0,min(h-1,(int)floorf(fmaxf(t[1],fmaxf(t[3],t[5]))*h)));
}
extern "C" __global__ void mesh_uv_bin_count(const float* uv,long long* counts,int start,int count,int w,int h) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int f=start+local,x0,y0,x1,y1;
    uv_bounds(uv+6LL*f,w,h,x0,y0,x1,y1);
    counts[f]=(long long)(x1-x0+1)*(y1-y0+1);
}
extern "C" __global__ void mesh_uv_bins(const float* uv,const long long* ends,unsigned long long* keys,
                                       int start,int count,int w,int h) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int f=start+local,x0,y0,x1,y1;
    uv_bounds(uv+6LL*f,w,h,x0,y0,x1,y1);
    long long output=f?ends[f-1]:0;
    for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x)
        keys[output++]=((unsigned long long)((long long)y*w+x)<<32)|(unsigned int)f;
}
__device__ bool uv_interiors_overlap(const float* a,const float* b) {
    // Separating axes detect even sub-texel overlap; touching edges remain valid.
    float area_a=(a[2]-a[0])*(a[5]-a[1])-(a[3]-a[1])*(a[4]-a[0]);
    float area_b=(b[2]-b[0])*(b[5]-b[1])-(b[3]-b[1])*(b[4]-b[0]);
    // FP32 UVs can collapse sub-texel faces to points; zero-area interiors cannot overlap.
    if(area_a==0.f || area_b==0.f) return false;
    for(int side=0;side<2;++side) {
        const float* t=side?b:a;
        for(int edge_id=0;edge_id<3;++edge_id) {
            int next=(edge_id+1)%3;
            float dx=t[2*next]-t[2*edge_id],dy=t[2*next+1]-t[2*edge_id+1];
            float epsilon=1.e-6f*(dx*dx+dy*dy);
            if(epsilon==0) continue;
            float amin=1.e30f,amax=-1.e30f,bmin=1.e30f,bmax=-1.e30f;
            for(int j=0;j<3;++j) {
                float pa=dx*(a[2*j+1]-t[2*edge_id+1])-dy*(a[2*j]-t[2*edge_id]);
                float pb=dx*(b[2*j+1]-t[2*edge_id+1])-dy*(b[2*j]-t[2*edge_id]);
                amin=fminf(amin,pa); amax=fmaxf(amax,pa); bmin=fminf(bmin,pb); bmax=fmaxf(bmax,pb);
            }
            if(fminf(amax,bmax)-fmaxf(amin,bmin)<=epsilon) return false;
        }
    }
    return true;
}
extern "C" __global__ void mesh_uv_overlap(const float* uv,const int* charts,const unsigned long long* keys,
    int* conflicts,int start,int count,long long total) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    long long entry=(long long)start+local;
    unsigned long long key=keys[entry],bin=key>>32;
    int face=(int)(unsigned int)key,chart=charts[face];
    if(conflicts[chart]) return;
    for(long long other=entry+1;other<total && (keys[other]>>32)==bin;++other) {
        int candidate=(int)(unsigned int)keys[other];
        if(candidate!=face && charts[candidate]==chart && uv_interiors_overlap(uv+6LL*face,uv+6LL*candidate)) {
            atomicExch(conflicts+chart,1); return;
        }
    }
}
extern "C" __global__ void mesh_edge_count(const float* uv,const int* charts,const unsigned long long* keys,
    long long* counts,int start,int count,long long total) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    long long entry=(long long)start+local,found=0;
    unsigned long long key=keys[entry],bin=key>>32;
    int f=(int)(unsigned int)key;
    for(long long other=entry+1;other<total && (keys[other]>>32)==bin;++other) {
        int candidate=(int)(unsigned int)keys[other];
        if(candidate!=f && charts[candidate]==charts[f] && uv_interiors_overlap(uv+6LL*f,uv+6LL*candidate)) ++found;
    }
    counts[entry]=found;
}
extern "C" __global__ void mesh_edge_write(const float* uv,const int* charts,const unsigned long long* keys,
    const long long* ends,unsigned long long* edges,int start,int count,long long total) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    long long entry=(long long)start+local,output=entry?ends[entry-1]:0;
    unsigned long long key=keys[entry],bin=key>>32;
    int f=(int)(unsigned int)key;
    for(long long other=entry+1;other<total && (keys[other]>>32)==bin;++other) {
        int candidate=(int)(unsigned int)keys[other];
        if(candidate!=f && charts[candidate]==charts[f] && uv_interiors_overlap(uv+6LL*f,uv+6LL*candidate))
            edges[output++]=((unsigned long long)(unsigned int)min(f,candidate)<<32)|(unsigned int)max(f,candidate);
    }
}
__device__ unsigned int priority(int face) {
    unsigned int value=(unsigned int)face;
    value^=value>>16; value*=0x7feb352dU; value^=value>>15; value*=0x846ca68bU; value^=value>>16;
    return value;
}
extern "C" __global__ void mesh_color_blocked(const unsigned long long* edges,const int* colors,
    const int* candidates,int* flags,int start,int count) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    unsigned long long edge_value=edges[(long long)start+local];
    int a=(int)(edge_value>>32),b=(int)(unsigned int)edge_value;
    if(colors[a]<0 && colors[b]>=0 && candidates[a]==colors[b]) atomicExch(flags+a,1);
    if(colors[b]<0 && colors[a]>=0 && candidates[b]==colors[a]) atomicExch(flags+b,1);
}
extern "C" __global__ void mesh_color_conflicts(const unsigned long long* edges,const int* colors,
    const int* candidates,int* flags,int start,int count) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    unsigned long long edge_value=edges[(long long)start+local];
    int a=(int)(edge_value>>32),b=(int)(unsigned int)edge_value;
    if(colors[a]<0 && colors[b]<0 && candidates[a]==candidates[b]) {
        unsigned int pa=priority(a),pb=priority(b);
        int loser=(pa<pb || (pa==pb && a<b))?a:b;
        atomicExch(flags+loser,1);
    }
}
extern "C" __global__ void mesh_uv_attributes(const float* vertices,const int* faces,const float* normals,
    const float* colors,const float* uv,const int* owner,float* positions,float* out_normals,float* out_colors,
    int start,int count,int w,int h) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int p=start+local,f=owner[p];
    if(f==2147483647) return;
    float a,b,c,t[6];
    for(int j=0;j<6;++j) t[j]=uv[6LL*f+j]*(j%2?h:w);
    bary(t,(p%w)+.5f,(p/w)+.5f,a,b,c);
    for(int j=0;j<3;++j) {
        int ia=faces[3LL*f],ib=faces[3LL*f+1],ic=faces[3LL*f+2];
        positions[3LL*p+j]=a*vertices[3LL*ia+j]+b*vertices[3LL*ib+j]+c*vertices[3LL*ic+j];
        out_normals[3LL*p+j]=a*normals[3LL*ia+j]+b*normals[3LL*ib+j]+c*normals[3LL*ic+j];
        out_colors[3LL*p+j]=a*colors[3LL*ia+j]+b*colors[3LL*ib+j]+c*colors[3LL*ic+j];
    }
}
extern "C" __global__ void mesh_texture_view(const float* positions,const float* normals,const int* owner,
    const float* matrix,const float* center,const float* reference,const float* mesh_depth,
    const unsigned char* alpha,const unsigned char* rgb,float* scores,float* colors,
    int start,int count,int w,int h,float voxel,float tolerance,float focal) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int p=start+local;
    if(owner[p]==2147483647) return;
    V3 point_value=point(positions,p),q=project(matrix,point_value);
    if(q.z<=0) return;
    float x=q.x/q.z-.5f,y=q.y/q.z-.5f;
    if(x<0 || y<0 || x>=w-1 || y>=h-1) return;
    int ix=(int)floorf(x),iy=(int)floorf(y),nx=(int)floorf(x+.5f),ny=(int)floorf(y+.5f);
    long long i=(long long)iy*w+ix,nearest=(long long)ny*w+nx;
    if(min(min(alpha[i],alpha[i+1]),min(alpha[i+w],alpha[i+w+1]))<243) return;
    float observed=reference[nearest],hit=mesh_depth[nearest];
    if(observed<=0 || !isfinite(hit) || fabsf(observed-q.z)>fmaxf(4*voxel,tolerance*q.z)) return;
    V3 direction={point_value.x-center[0],point_value.y-center[1],point_value.z-center[2]};
    float distance=sqrtf(dot(direction,direction));
    if(fabsf(hit/q.z-1.f)*distance>2*voxel) return;
    V3 n=point(normals,p);
    float cosine=fabsf(dot(n,direction))/fmaxf(distance*sqrtf(dot(n,n)),1.e-12f);
    float score=cosine*cosine*cosine*cosine*focal/(q.z*q.z);
    if(cosine<=.1f || score<=scores[p]) return;
    float fx=x-ix,fy=y-iy;
    for(int j=0;j<3;++j) colors[3LL*p+j]=(
        (1-fy)*((1-fx)*rgb[3*i+j]+fx*rgb[3*(i+1)+j])+
        fy*((1-fx)*rgb[3*(i+w)+j]+fx*rgb[3*(i+w+1)+j]))/255.f;
    scores[p]=score;
}
extern "C" __global__ void mesh_gutter(const int* source,int* target,int start,int count,int w,int h,int radius) {
    int local=blockIdx.x*blockDim.x+threadIdx.x;
    if(local>=count) return;
    int p=start+local,x=p%w,y=p/w,best=source[p];
    int distance=best<0?2147483647:(best%w-x)*(best%w-x)+(best/w-y)*(best/w-y);
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        int xx=x+dx,yy=y+dy;
        if(xx<0 || yy<0 || xx>=w || yy>=h) continue;
        int seed=source[(long long)yy*w+xx];
        if(seed<0) continue;
        int d=(seed%w-x)*(seed%w-x)+(seed/w-y)*(seed/w-y);
        if(d<=radius*radius && (d<distance || (d==distance && seed<best))) { best=seed; distance=d; }
    }
    target[p]=best;
}
'''


@lru_cache(maxsize=None)
def _kernels(device_index):
    with torch.cuda.device(device_index):
        _context = torch.empty(1, device=torch.device("cuda", device_index))
        return dict(zip(
            ("depth", "choose_view", "connect", "compress", "uv_raster", "uv_bin_count", "uv_bins", "uv_overlap",
             "edge_count", "edge_write", "color_blocked", "color_conflicts", "uv_attributes", "texture_view", "gutter"),
            load_surface_kernels(
                _SOURCE, ("mesh_depth", "mesh_choose_view", "mesh_connect", "mesh_compress",
                          "mesh_uv_raster", "mesh_uv_bin_count", "mesh_uv_bins", "mesh_uv_overlap",
                          "mesh_edge_count", "mesh_edge_write", "mesh_color_blocked", "mesh_color_conflicts",
                          "mesh_uv_attributes", "mesh_texture_view", "mesh_gutter"),
                torch.__file__, torch.version.cuda, torch.cuda.get_device_capability(device_index),
            ),
        ))


def prepare_mesh_cuda(device):
    with torch.cuda.device(device):
        _kernels(torch.cuda.current_device())


def launch(name, tensors, count, scalars, interrupt):
    """Bound launch duration, retain integer face IDs, and honor cancellation between chunks."""
    if count < 0 or count > 0x7FFFFFFF:
        raise ValueError("Mesh GPU worklist exceeds the kernels' 32-bit entry addressing")
    device = tensors[0].device
    with torch.cuda.device(device):
        kernel = _kernels(torch.cuda.current_device())[name]
        pointers = tuple(C.c_void_p(t.data_ptr()) for t in tensors)
        stream = torch.cuda.current_stream(device).cuda_stream
        for start in range(0, count, 262_144):
            interrupt()
            length = min(262_144, count - start)
            kernel(((length + 255) // 256, 1), (256, 1),
                   pointers + (C.c_int(start), C.c_int(length)) + tuple(scalars), stream=stream)


def depth_map(vertices, faces, matrix, width, height, interrupt):
    depth = torch.full((height, width), float("inf"), device=vertices.device)
    launch("depth", (vertices, faces, matrix, depth), len(faces),
           (C.c_int(width), C.c_int(height)), interrupt)
    return depth


def connected_charts(faces, groups, interrupt):
    """Connect only touching faces with the same projection and spatial partition."""
    count = len(faces)
    capacity = 1 << max(1, (max(2, count * 4) - 1).bit_length())
    keys = torch.full((capacity,), -1, device=faces.device, dtype=torch.int64)
    values = torch.full((capacity,), -1, device=faces.device, dtype=torch.int32)
    parent = torch.arange(count, device=faces.device, dtype=torch.int32)
    launch("connect", (faces, groups, keys, values, parent), count,
           (C.c_ulonglong(capacity - 1),), interrupt)
    launch("compress", (parent,), count, (), interrupt)
    del keys, values
    return torch.unique(parent, sorted=True, return_inverse=True)[1].to(torch.int32)


def _uv_bins(uv, size, interrupt):
    counts = torch.empty(len(uv), device=uv.device, dtype=torch.int64)
    launch("uv_bin_count", (uv, counts), len(uv), (C.c_int(size), C.c_int(size)), interrupt)
    ends = counts.cumsum(0)
    total = int(ends[-1])
    if total > 0x7FFFFFFF:
        raise ValueError("UV spatial bins exceed this device path's addressable entry count")
    keys = torch.empty(total, device=uv.device, dtype=torch.int64)
    launch("uv_bins", (uv, ends, keys), len(uv), (C.c_int(size), C.c_int(size)), interrupt)
    del counts, ends
    return keys.sort().values


def uv_map(uv, charts, size, interrupt):
    owner = torch.full((size, size), 0x7FFFFFFF, device=uv.device, dtype=torch.int32)
    conflicts = torch.zeros(int(charts.max()) + 1, device=uv.device, dtype=torch.int32)
    launch("uv_raster", (uv, owner), len(uv),
           (C.c_int(size), C.c_int(size)), interrupt)
    keys = _uv_bins(uv, size, interrupt)
    total = len(keys)
    launch("uv_overlap", (uv, charts, keys, conflicts), total, (C.c_longlong(total),), interrupt)
    return owner, conflicts


def color_overlaps(uv, charts, size, interrupt):
    """Color the actual intersection graph, not arbitrary depth slices of a curved surface."""
    keys = _uv_bins(uv, size, interrupt)
    counts = torch.empty(len(keys), device=uv.device, dtype=torch.int64)
    launch("edge_count", (uv, charts, keys, counts), len(keys), (C.c_longlong(len(keys)),), interrupt)
    ends = counts.cumsum(0)
    total = int(ends[-1])
    del counts
    edges = torch.empty(total, device=uv.device, dtype=torch.int64)
    launch("edge_write", (uv, charts, keys, ends, edges), len(keys), (C.c_longlong(len(keys)),), interrupt)
    del keys, ends
    edges = torch.unique(edges, sorted=True)
    print(f"[VNCCS PLY2MESH] GPU intersection graph: {len(edges)} conflicting triangle pairs", flush=True)
    return _color_graph(edges, len(uv), interrupt)


def _color_graph(edges, count, interrupt):
    colors = torch.full((count,), -1, device=edges.device, dtype=torch.int32)
    flags = torch.zeros_like(colors)
    iteration = 0
    while True:
        interrupt()
        iteration += 1
        unassigned = colors < 0
        candidates = torch.zeros_like(colors)
        # Pick the first free color against committed neighbors, without a fixed layer limit.
        while True:
            flags.zero_()
            launch("color_blocked", (edges, colors, candidates, flags), len(edges), (), interrupt)
            if not torch.count_nonzero(flags):
                break
            candidates.add_(flags)
        flags.zero_()
        launch("color_conflicts", (edges, colors, candidates, flags), len(edges), (), interrupt)
        colors = torch.where(unassigned & (flags == 0), candidates, colors)
        pending = int(torch.count_nonzero(colors < 0))
        print(f"[VNCCS PLY2MESH] GPU UV layer coloring {iteration}: pending={pending}, "
              f"layers={int(colors.max()) + 1}", flush=True)
        if not pending:
            return colors


def dilate_texture(colors, owner, size, interrupt, radius=4):
    valid = owner.reshape(-1) != 0x7FFFFFFF
    seeds = torch.arange(size * size, device=owner.device, dtype=torch.int32)
    seeds[~valid] = -1
    target = torch.empty_like(seeds)
    for _ in range(radius):
        launch("gutter", (seeds, target), len(seeds),
               (C.c_int(size), C.c_int(size), C.c_int(radius)), interrupt)
        seeds, target = target, seeds
    result = torch.zeros_like(colors)
    valid = seeds >= 0
    result[valid] = colors[seeds[valid].long()]
    return result

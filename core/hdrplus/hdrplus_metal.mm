// GPU (Metal) implementation of the HDR+ pipeline — the fast, low-RAM path the
// HDR toggle wants. It mirrors the CPU twin in core/hdrplus/hdrplus.cpp stage
// for stage (same integer domain, casts, floor-division, mirror/clamp borders).
//
// The kernels are RUNTIME-COMPILED MSL (newLibraryWithSource). That is a
// deliberate safety choice: if the kernel source fails to compile, or Metal is
// unavailable, process_gpu() returns false and hdrplus::process() falls back to
// the CPU twin. A kernel error therefore degrades to "slower" rather than
// breaking the whole app build (which a .metal file linked into the shared
// metallib would).
//
// DEVICE-UNVERIFIED: there is no Metal toolchain in the authoring environment,
// so this has not been compiled or run. The CPU twin is the parity reference —
// each kernel is a line-by-line port of the matching CPU function. Validate on
// device against the CPU path (process(..., allow_gpu:false)) before trusting.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "hdrplus.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <utility>

namespace hhsr { namespace hdrplus {

namespace {

// ===========================================================================
// MSL kernel source. Helpers mirror hdrplus.cpp's numeric semantics exactly.
// ===========================================================================
static const char kMSLsrc[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline int fdiv(int a, int b){ int q=a/b, r=a%b; if(r!=0 && ((r<0)!=(b<0))) q-=1; return q; }
inline int fmod_(int a, int b){ int r=a%b; if(r!=0 && ((r<0)!=(b<0))) r+=b; return r; }
inline int mirn(int i, int n){ if(n<=1) return 0; int p=2*(n-1); int m=((i%p)+p)%p; return m<n?m:p-m; }
inline int clmpn(int i, int n){ return i<0?0:(i>=n?n-1:i); }
inline ushort u16c(float x){ return (ushort)(int)x; }                       // trunc + wrap16
inline ushort u16s(float x){ int v=(int)x; return (ushort)(v<0?0:(v>65535?65535:v)); }
inline uchar  u8s (float x){ int v=(int)x; return (uchar)(v<0?0:(v>255?255:v)); }
inline ushort mget(device const ushort* p, int x, int y, int w, int h){
    if((uint)x<(uint)w && (uint)y<(uint)h) return p[(uint)y*w+x];
    return p[(uint)mirn(y,h)*w + mirn(x,w)];
}

// ---- downsamplers ----------------------------------------------------------
// dim = (sw,sh,dw,dh)
kernel void k_box_down2(device const ushort* src [[buffer(0)]],
                        device ushort* dst       [[buffer(1)]],
                        constant int4& dim       [[buffer(2)]],
                        uint2 g [[thread_position_in_grid]]) {
    int dw=dim.z, dh=dim.w; if((int)g.x>=dw||(int)g.y>=dh) return;
    int sw=dim.x, sh=dim.y; int x=g.x, y=g.y;
    uint s = (uint)mget(src,2*x,2*y,sw,sh)+mget(src,2*x+1,2*y,sw,sh)
           + mget(src,2*x,2*y+1,sw,sh)+mget(src,2*x+1,2*y+1,sw,sh);
    dst[(uint)y*dw+x] = u16c((float)(s/4u));
}
kernel void k_gauss_down4(device const ushort* src [[buffer(0)]],
                          device ushort* dst       [[buffer(1)]],
                          constant int4& dim       [[buffer(2)]],
                          uint2 g [[thread_position_in_grid]]) {
    int dw=dim.z, dh=dim.w; if((int)g.x>=dw||(int)g.y>=dh) return;
    int sw=dim.x, sh=dim.y; int x=g.x, y=g.y;
    const int k[5][5]={{2,4,5,4,2},{4,9,12,9,4},{5,12,15,12,5},{4,9,12,9,4},{2,4,5,4,2}};
    uint s=0;
    for(int ry=-2;ry<=2;++ry) for(int rx=-2;rx<=2;++rx)
        s += (uint)mget(src,4*x+rx,4*y+ry,sw,sh)*(uint)k[ry+2][rx+2];
    dst[(uint)y*dw+x]=u16c((float)(s/159u));
}

// ---- align_layer -----------------------------------------------------------
// P: dw,dh (layer, same for ref/alt), outx0,outy0,outnx,outny,
//    prevx0,prevy0,prevnx,prevny, pmin,pmax, prevConst
struct AlignP { int dw,dh, ox0,oy0,onx,ony, px0,py0,pnx,pny, pmin,pmax, prevConst, _pad; };
kernel void k_align(device const ushort* layRef [[buffer(0)]],
                    device const ushort* layN   [[buffer(1)]],
                    device const int2* prevA     [[buffer(2)]],
                    device int2* outA            [[buffer(3)]],
                    constant AlignP& P           [[buffer(4)]],
                    uint2 g [[thread_position_in_grid]]) {
    if((int)g.x>=P.onx||(int)g.y>=P.ony) return;
    int tx=P.ox0+(int)g.x, ty=P.oy0+(int)g.y;
    int pox=0, poy=0;
    if(!P.prevConst){
        int ptx=fdiv(tx-1,4), pty=fdiv(ty-1,4);
        int ix=ptx-P.px0, iy=pty-P.py0;
        ix = ix<0?0:(ix>=P.pnx?P.pnx-1:ix);
        iy = iy<0?0:(iy>=P.pny?P.pny-1:iy);
        int2 pv = prevA[(uint)iy*P.pnx+ix];
        int cvx = max(P.pmin, min(P.pmax, pv.x));
        int cvy = max(P.pmin, min(P.pmax, pv.y));
        pox = 4*cvx; poy = 4*cvy;
    }
    int best = 0x7fffffff; int bxi=-4, byi=-4;
    for(int yi=-4; yi<=3; ++yi){
        for(int xi=-4; xi<=3; ++xi){
            int score=0;
            for(int ry=0; ry<16; ++ry){
                int y0=ty*8+ry; int ay=y0+poy+yi;
                for(int rx=0; rx<16; ++rx){
                    int x0=tx*8+rx; int ax=x0+pox+xi;
                    int rv=mget(layRef,x0,y0,P.dw,P.dh);
                    int av=mget(layN,ax,ay,P.dw,P.dh);
                    score += abs(rv-av);
                }
            }
            if(score<best){ best=score; bxi=xi; byi=yi; }
        }
    }
    outA[(uint)g.y*P.onx+g.x] = int2(bxi+pox, byi+poy);
}

// ---- temporal weight (per comp) --------------------------------------------
// P: num_tx,num_ty, TX0,TY0,TNX,TNY, dwL,dhL
struct WP { int num_tx,num_ty, TX0,TY0,TNX,TNY, dwL,dhL; };
kernel void k_weight(device const ushort* layRef [[buffer(0)]],
                     device const ushort* layN   [[buffer(1)]],
                     device const int2* a0        [[buffer(2)]], // level-0 offsets, num_tx*num_ty
                     device float* wout           [[buffer(3)]], // TNX*TNY
                     constant WP& P               [[buffer(4)]],
                     uint2 g [[thread_position_in_grid]]) {
    if((int)g.x>=P.TNX||(int)g.y>=P.TNY) return;
    int tx=P.TX0+(int)g.x, ty=P.TY0+(int)g.y;
    int ctx=clmpn(tx,P.num_tx), cty=clmpn(ty,P.num_ty);
    int2 a = a0[(uint)cty*P.num_tx+ctx];
    int ox = max(-168, min(126, 2*a.x));
    int oy = max(-168, min(126, 2*a.y));
    int ohx=fdiv(ox,2), ohy=fdiv(oy,2);
    int sumL1=0;
    for(int ry=0;ry<16;++ry){ int ly=ty*8+ry; int aly=ly+ohy;
        for(int rx=0;rx<16;++rx){ int lx=tx*8+rx; int alx=lx+ohx;
            int rv=mget(layRef,lx,ly,P.dwL,P.dhL);
            int av=mget(layN,alx,aly,P.dwL,P.dhL);
            sumL1 += abs(rv-av);
        }}
    int dist=(int)(sumL1/256);
    float norm_dist = max(1.0f, (float)dist/8.0f - 10.0f/8.0f);
    wout[(uint)g.y*P.TNX+g.x] = (norm_dist > (300.0f-10.0f)) ? 0.0f : (1.0f/norm_dist);
}

// total = 1 + sum over comps
// dim = (ncomp, TNXTNY)
kernel void k_total(device const float* wAll [[buffer(0)]],
                    device float* total      [[buffer(1)]],
                    constant int2& dim       [[buffer(2)]],
                    uint gi [[thread_position_in_grid]]) {
    if((int)gi>=dim.y) return;
    float s=1.0f;
    for(int c=0;c<dim.x;++c) s += wAll[(uint)c*dim.y+gi];
    total[gi]=s;
}

// ---- spatial merge (temporal folded in) ------------------------------------
// P: w,h, num_tx,num_ty, TX0,TY0,TNX,TNY, N
struct MP { int w,h, num_tx,num_ty, TX0,TY0,TNX,TNY, N, _p0,_p1; };
inline float cosw32(int v){ return 0.5f - 0.5f*cos(2.0f*3.141592f*((float)v+0.5f)/32.0f); }
kernel void k_spatial(device const ushort* rawAll [[buffer(0)]], // N*w*h
                      device const int2* a0All     [[buffer(1)]], // (N-1)*num_tx*num_ty
                      device const float* wAll      [[buffer(2)]], // (N-1)*TNX*TNY
                      device const float* total     [[buffer(3)]], // TNX*TNY
                      device ushort* merged         [[buffer(4)]], // w*h
                      constant MP& P                [[buffer(5)]],
                      uint2 g [[thread_position_in_grid]]) {
    int w=P.w,h=P.h; if((int)g.x>=w||(int)g.y>=h) return;
    int x=g.x, y=g.y;
    int tx0=fdiv(x,16)-1, tx1=fdiv(x,16);
    int ty0=fdiv(y,16)-1, ty1=fdiv(y,16);
    int ix0=fmod_(x,16)+16, ix1=fmod_(x,16);
    int iy0=fmod_(y,16)+16, iy1=fmod_(y,16);
    int4 txv=int4(tx0,tx1,tx0,tx1), tyv=int4(ty0,ty0,ty1,ty1);
    int4 ixv=int4(ix0,ix1,ix0,ix1), iyv=int4(iy0,iy0,iy1,iy1);
    int ntile=P.num_tx*P.num_ty, ntileW=P.TNX*P.TNY;
    float out=0.0f;
    for(int t=0;t<4;++t){
        int tx=txv[t], ty=tyv[t], ix=ixv[t], iy=iyv[t];
        float wgt=cosw32(ix)*cosw32(iy);
        if(wgt==0.0f) continue;
        int wtile=(ty-P.TY0)*P.TNX+(tx-P.TX0);
        float tot=total[wtile];
        int rxi=tx*16+ix, ryi=ty*16+iy;
        float tv = (float)mget(rawAll, rxi, ryi, w, h) / tot; // ref = frame 0
        int ctx=clmpn(tx,P.num_tx), cty=clmpn(ty,P.num_ty);
        for(int c=1;c<P.N;++c){
            int slot=c-1;
            int2 a = a0All[(uint)slot*ntile + (uint)cty*P.num_tx+ctx];
            int ox=2*a.x, oy=2*a.y;
            int av = mget(rawAll + (uint)c*w*h, rxi+ox, ryi+oy, w, h);
            tv += wAll[(uint)slot*ntileW + wtile] * (float)av / tot;
        }
        out += wgt*tv;
    }
    merged[(uint)y*w+x] = u16c(out);
}

// ===================== finish ==============================================
// FP: w,h, sox,soy, bp, wp (as ints), and wb gains
struct FP { int w,h,sox,soy,bp,wp; float wbr,wbg0,wbg1,wbb; };
kernel void k_bwl_wb(device const ushort* merged [[buffer(0)]],
                     device ushort* wb            [[buffer(1)]],
                     constant FP& P               [[buffer(2)]],
                     uint2 g [[thread_position_in_grid]]) {
    if((int)g.x>=P.w||(int)g.y>=P.h) return;
    int x=g.x,y=g.y;
    int raw = mget(merged, x+P.sox, y+P.soy, P.w, P.h);
    float wf = 65535.0f/((float)P.wp-(float)P.bp);
    float lvl = (float)u16s(((float)raw-(float)P.bp)*wf);
    float gg = ((y&1)==0) ? (((x&1)==0)?P.wbr:P.wbg0) : (((x&1)==0)?P.wbg1:P.wbb);
    wb[(uint)y*P.w+x]=u16s(gg*lvl);
}

// Malvar demosaic. out = [w*h*3] interleaved. dim=(w,h)
kernel void k_demosaic(device const ushort* wb [[buffer(0)]],
                       device ushort* rgb       [[buffer(1)]],
                       constant int2& dim       [[buffer(2)]],
                       uint2 g [[thread_position_in_grid]]) {
    int w=dim.x,h=dim.y; if((int)g.x>=w||(int)g.y>=h) return;
    int x=g.x,y=g.y;
    const int f0[5][5]={{0,0,-1,0,0},{0,0,2,0,0},{-1,2,4,2,-1},{0,0,2,0,0},{0,0,-1,0,0}};
    const int f1[5][5]={{0,0,1,0,0},{0,-2,0,-2,0},{-2,8,10,8,-2},{0,-2,0,-2,0},{0,0,1,0,0}};
    const int f2[5][5]={{0,0,-2,0,0},{0,-2,8,-2,0},{1,0,10,0,1},{0,-2,8,-2,0},{0,0,-2,0,0}};
    const int f3[5][5]={{0,0,-3,0,0},{0,4,0,4,0},{-3,0,12,0,-3},{0,4,0,4,0},{0,0,-3,0,0}};
    int a0=0,a1=0,a2=0,a3=0;
    for(int ry=-2;ry<=2;++ry) for(int rx=-2;rx<=2;++rx){
        int v=mget(wb,x+rx,y+ry,w,h);
        a0+=v*f0[ry+2][rx+2]; a1+=v*f1[ry+2][rx+2]; a2+=v*f2[ry+2][rx+2]; a3+=v*f3[ry+2][rx+2];
    }
    ushort d0=u16s((float)a0/8.0f), d1=u16s((float)a1/16.0f), d2=u16s((float)a2/16.0f), d3=u16s((float)a3/16.0f);
    ushort self=wb[(uint)y*w+x];
    bool Rr_=(y%2==0), Rc=(x%2==0); bool Br_=!Rr_, Bc=!Rc;
    ushort R,G,B;
    if(Rr_&&Rc) R=self; else if(Rr_&&Bc) R=d1; else if(Br_&&Rc) R=d2; else R=d3;
    if((Rr_&&Rc)||(Br_&&Bc)) G=d0; else G=self;
    if(Br_&&Bc) B=self; else if(Br_&&Rc) B=d1; else if(Rr_&&Bc) B=d2; else B=d3;
    uint o=((uint)y*w+x)*3; rgb[o]=R; rgb[o+1]=G; rgb[o+2]=B;
}

// srgb: out_c = sum_r ccm[r*3+c]*in_r. dim=(w,h). ccm 9 floats.
kernel void k_srgb(device const ushort* rgb [[buffer(0)]],
                   device ushort* out        [[buffer(1)]],
                   constant int2& dim        [[buffer(2)]],
                   constant float* ccm       [[buffer(3)]],
                   uint gi [[thread_position_in_grid]]) {
    uint n=(uint)dim.x*dim.y; if(gi>=n) return;
    float in0=rgb[gi*3], in1=rgb[gi*3+1], in2=rgb[gi*3+2];
    for(int c=0;c<3;++c)
        out[gi*3+c]=u16s(ccm[0*3+c]*in0+ccm[1*3+c]*in1+ccm[2*3+c]*in2);
}

kernel void k_gray(device const ushort* srgb [[buffer(0)]],
                   device ushort* gray        [[buffer(1)]],
                   constant int2& dim         [[buffer(2)]],
                   uint gi [[thread_position_in_grid]]) {
    uint n=(uint)dim.x*dim.y; if(gi>=n) return;
    gray[gi]=u16c((float)(((uint)srgb[gi*3]+srgb[gi*3+1]+srgb[gi*3+2])/3u));
}

// gamma correct / inverse, single channel (also used over 3ch flattened).
kernel void k_gamma_correct(device const ushort* in [[buffer(0)]],
                            device ushort* out        [[buffer(1)]],
                            constant uint& n          [[buffer(2)]],
                            uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return; int v=in[gi];
    out[gi] = (v<200) ? u16c(12.92f*v) : u16c(680.552897f*pow((float)v,0.416667f)-3604.425f);
}
kernel void k_gamma_inverse(device const ushort* in [[buffer(0)]],
                            device ushort* out        [[buffer(1)]],
                            constant uint& n          [[buffer(2)]],
                            uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return; int v=in[gi];
    out[gi] = (v<2575) ? u16c(0.0774f*v) : u16c(pow((float)v/65535.0f+0.055f,2.4f)*57632.49226f);
}
// out = u16s(gain * in)
kernel void k_brighten(device const ushort* in [[buffer(0)]],
                       device ushort* out        [[buffer(1)]],
                       constant float& gain      [[buffer(2)]],
                       constant uint& n          [[buffer(3)]],
                       uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return; out[gi]=u16s(gain*(float)(uint)in[gi]);
}

// gauss7 separable, repeat_edge. dim=(w,h). bx is float scratch (w*h).
kernel void k_g7x_u16(device const ushort* in [[buffer(0)]],
                      device float* bx          [[buffer(1)]],
                      constant int2& dim        [[buffer(2)]],
                      uint2 g [[thread_position_in_grid]]) {
    int w=dim.x,h=dim.y; if((int)g.x>=w||(int)g.y>=h) return;
    const float K[7]={0.026267f,0.100742f,0.225511f,0.29496f,0.225511f,0.100742f,0.026267f};
    float s=0; for(int r=-3;r<=3;++r) s+=(float)in[(uint)g.y*w+clmpn((int)g.x+r,w)]*K[r+3];
    bx[(uint)g.y*w+g.x]=s;
}
kernel void k_g7x_f(device const float* in [[buffer(0)]],
                    device float* bx        [[buffer(1)]],
                    constant int2& dim      [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    int w=dim.x,h=dim.y; if((int)g.x>=w||(int)g.y>=h) return;
    const float K[7]={0.026267f,0.100742f,0.225511f,0.29496f,0.225511f,0.100742f,0.026267f};
    float s=0; for(int r=-3;r<=3;++r) s+=in[(uint)g.y*w+clmpn((int)g.x+r,w)]*K[r+3];
    bx[(uint)g.y*w+g.x]=s;
}
kernel void k_g7y_u16(device const float* bx [[buffer(0)]],
                      device ushort* out      [[buffer(1)]],
                      constant int2& dim      [[buffer(2)]],
                      uint2 g [[thread_position_in_grid]]) {
    int w=dim.x,h=dim.y; if((int)g.x>=w||(int)g.y>=h) return;
    const float K[7]={0.026267f,0.100742f,0.225511f,0.29496f,0.225511f,0.100742f,0.026267f};
    float s=0; for(int r=-3;r<=3;++r) s+=bx[(uint)clmpn((int)g.y+r,h)*w+g.x]*K[r+3];
    out[(uint)g.y*w+g.x]=u16c(s);
}
kernel void k_g7y_f(device const float* bx [[buffer(0)]],
                    device float* out       [[buffer(1)]],
                    constant int2& dim      [[buffer(2)]],
                    uint2 g [[thread_position_in_grid]]) {
    int w=dim.x,h=dim.y; if((int)g.x>=w||(int)g.y>=h) return;
    const float K[7]={0.026267f,0.100742f,0.225511f,0.29496f,0.225511f,0.100742f,0.026267f};
    float s=0; for(int r=-3;r<=3;++r) s+=bx[(uint)clmpn((int)g.y+r,h)*w+g.x]*K[r+3];
    out[(uint)g.y*w+g.x]=s;
}

// combine helpers
kernel void k_mask_init(device const ushort* im1 [[buffer(0)]],
                        device const ushort* im2  [[buffer(1)]],
                        device const float* ndist [[buffer(2)]],
                        device float* mask1        [[buffer(3)]],
                        device float* mask2        [[buffer(4)]],
                        constant uint& n           [[buffer(5)]],
                        uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return;
    float w1=ndist[im1[gi]], w2=ndist[im2[gi]];
    float m1=w1/(w1+w2); mask1[gi]=m1; mask2[gi]=1.0f-m1;
}
// accum = i32(lap1*mask1) + i32(lap2*mask2), lap = unb - bl
kernel void k_band(device const ushort* unb1 [[buffer(0)]],
                   device const ushort* bl1   [[buffer(1)]],
                   device const ushort* unb2  [[buffer(2)]],
                   device const ushort* bl2   [[buffer(3)]],
                   device const float* mask1  [[buffer(4)]],
                   device const float* mask2  [[buffer(5)]],
                   device int* accum          [[buffer(6)]],
                   constant uint& n           [[buffer(7)]],
                   uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return;
    int lap1=(int)unb1[gi]-(int)bl1[gi], lap2=(int)unb2[gi]-(int)bl2[gi];
    accum[gi] = (int)((float)lap1*mask1[gi]) + (int)((float)lap2*mask2[gi]);
}
kernel void k_residual(device const int* accum [[buffer(0)]],
                       device const ushort* bl1 [[buffer(1)]],
                       device const float* mask1 [[buffer(2)]],
                       device const ushort* bl2  [[buffer(3)]],
                       device const float* mask2 [[buffer(4)]],
                       device ushort* out         [[buffer(5)]],
                       constant uint& n           [[buffer(6)]],
                       uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return;
    int a = accum[gi] + (int)((float)bl1[gi]*mask1[gi]) + (int)((float)bl2[gi]*mask2[gi]);
    out[gi]=u16s((float)a);
}
// recolor: toned_c = u16s(srgb_c * dark / max(1,gray))
kernel void k_recolor(device const ushort* srgb [[buffer(0)]],
                      device const ushort* dark  [[buffer(1)]],
                      device const ushort* gray  [[buffer(2)]],
                      device ushort* toned        [[buffer(3)]],
                      constant uint& n            [[buffer(4)]],
                      uint gi [[thread_position_in_grid]]) {
    if(gi>=n) return;
    uint gy = max(1u,(uint)gray[gi]);
    for(int c=0;c<3;++c)
        toned[gi*3+c]=u16s((float)((uint)srgb[gi*3+c]*(uint)dark[gi]/gy));
}
// contrast + black-level + u8/256 (per channel). CP carries precomputed consts.
struct CP { float slope, inner, constant_, factor, white_scale; int black_level; uint n; int _p; };
kernel void k_contrast_u8(device const ushort* toned [[buffer(0)]],
                          device uchar* out            [[buffer(1)]],
                          constant CP& P               [[buffer(2)]],
                          uint gi [[thread_position_in_grid]]) {
    if(gi>=P.n) return;
    float val = P.factor*(float)toned[gi];
    ushort s = u16s(P.slope*sin(val-P.inner)+P.constant_);
    ushort c2 = u16s(((float)(int)s-(float)P.black_level)*P.white_scale);
    out[gi]=u8s((float)(c2/256));
}
)METAL";

// ===========================================================================
// Driver
// ===========================================================================
// Plain PODs for the small int-vector params (match Metal int4/int2 byte
// layout: 4/2 contiguous int32s), so we need no <simd/simd.h> host types.
struct i4 { int x, y, z, w; };
struct i2 { int x, y; };

struct GpuCtx {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    id<MTLLibrary> lib = nil;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* psCache = nil;

    bool init() {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return false;
        q = [dev newCommandQueue];
        NSError* err = nil;
        MTLCompileOptions* opt = [MTLCompileOptions new];
        lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kMSLsrc] options:opt error:&err];
        if (!lib) { NSLog(@"[hdrplus_metal] MSL compile failed: %@", err); return false; }
        psCache = [NSMutableDictionary dictionary];
        return q != nil;
    }
    id<MTLComputePipelineState> ps(NSString* name) {
        id<MTLComputePipelineState> p = psCache[name];
        if (p) return p;
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) return nil;
        NSError* err = nil;
        p = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!p) { NSLog(@"[hdrplus_metal] pipeline %@ failed: %@", name, err); return nil; }
        psCache[name] = p;
        return p;
    }
    id<MTLBuffer> buf(size_t bytes) {
        return [dev newBufferWithLength:std::max<size_t>(bytes,16)
                                options:MTLResourceStorageModeShared];
    }
};

// Run one kernel as its own command buffer (serialized via waitUntilCompleted).
// `bufs` are device buffers bound at indices 0..; `bytesList` are small inline
// param blobs bound at the indices given. gx/gy are the grid dims (gy==1 => 1D).
struct Bind { id<MTLBuffer> b; };
static bool run1(GpuCtx& C, NSString* name,
                 NSArray<id<MTLBuffer>>* bufs,
                 const std::vector<std::pair<int, std::pair<const void*, size_t>>>& bytesAt,
                 int gx, int gy) {
    id<MTLComputePipelineState> p = C.ps(name);
    if (!p) return false;
    id<MTLCommandBuffer> cb = [C.q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:p];
    for (NSUInteger i = 0; i < bufs.count; ++i)
        if ((id)bufs[i] != (id)NSNull.null) [enc setBuffer:bufs[i] offset:0 atIndex:i];
    for (auto& ba : bytesAt) [enc setBytes:ba.second.first length:ba.second.second atIndex:ba.first];
    MTLSize grid = MTLSizeMake((NSUInteger)gx, (NSUInteger)std::max(1, gy), 1);
    NSUInteger tgx = (gy <= 1) ? std::min<NSUInteger>(256, p.maxTotalThreadsPerThreadgroup) : 16;
    NSUInteger tgy = (gy <= 1) ? 1 : 16;
    MTLSize tg = MTLSizeMake(tgx, tgy, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    return cb.error == nil;
}

} // namespace

bool process_gpu(const std::vector<const uint16_t*>& frames, int w, int h,
                 const Params& p,
                 std::vector<uint16_t>& merged_out,
                 std::vector<uint8_t>& rgb_out) {
    @autoreleasepool {
    const int N = (int)frames.size();
    if (N < 2 || w < 32 || h < 32) return false;
    const int num_tx = w/16 - 1, num_ty = h/16 - 1;
    if (num_tx <= 0 || num_ty <= 0) return false;

    GpuCtx C;
    if (!C.init()) return false;

    // prev_tile(t)=floor((t-1)/4).
    auto prev_tile = [](int t){ int a=t-1,b=4; int q=a/b,r=a%b; if(r!=0&&((r<0)!=(b<0)))q-=1; return q; };
    const int L1x = prev_tile(num_tx-1), L1y = prev_tile(num_ty-1);
    const int L2x = prev_tile(L1x),       L2y = prev_tile(L1y);
    const int a1nx = L1x-(-1)+1, a1ny = L1y-(-1)+1;
    const int a2nx = L2x-(-1)+1, a2ny = L2y-(-1)+1;

    const int w0=w/2,h0=h/2, w1=w0/4,h1=h0/4, w2=w1/4,h2=h1/4;
    if (w2 < 1 || h2 < 1) return false;

    // Upload raw frames (N*w*h u16).
    id<MTLBuffer> rawAll = C.buf((size_t)N*w*h*2);
    for (int n = 0; n < N; ++n)
        std::memcpy((uint8_t*)rawAll.contents + (size_t)n*w*h*2, frames[n], (size_t)w*h*2);

    // Reference pyramid.
    auto down2 = [&](id<MTLBuffer> src,int sw,int sh, id<MTLBuffer> dst,int dw,int dh)->bool{
        i4 dim{sw,sh,dw,dh};
        return run1(C,@"k_box_down2",@[src,dst],{{2,{&dim,sizeof(dim)}}},dw,dh);
    };
    auto gdown4 = [&](id<MTLBuffer> src,int sw,int sh, id<MTLBuffer> dst,int dw,int dh)->bool{
        i4 dim{sw,sh,dw,dh};
        return run1(C,@"k_gauss_down4",@[src,dst],{{2,{&dim,sizeof(dim)}}},dw,dh);
    };

    id<MTLBuffer> refL0=C.buf((size_t)w0*h0*2), refL1=C.buf((size_t)w1*h1*2), refL2=C.buf((size_t)w2*h2*2);
    {
        // reference raw is the first w*h slice of rawAll.
        id<MTLBuffer> ref0 = C.buf((size_t)w*h*2);
        std::memcpy(ref0.contents, frames[0], (size_t)w*h*2);
        if(!down2(ref0,w,h,refL0,w0,h0)) return false;
        if(!gdown4(refL0,w0,h0,refL1,w1,h1)) return false;
        if(!gdown4(refL1,w1,h1,refL2,w2,h2)) return false;
    }

    const int TX0=-1,TY0=-1, TNX=num_tx-TX0+1, TNY=num_ty-TY0+1;
    id<MTLBuffer> a0All = C.buf((size_t)(N-1)*num_tx*num_ty*sizeof(int)*2);
    id<MTLBuffer> wAll  = C.buf((size_t)(N-1)*TNX*TNY*sizeof(float));

    // Per-comp scratch.
    id<MTLBuffer> cL0=C.buf((size_t)w0*h0*2), cL1=C.buf((size_t)w1*h1*2), cL2=C.buf((size_t)w2*h2*2);
    id<MTLBuffer> a2=C.buf((size_t)a2nx*a2ny*sizeof(int)*2), a1=C.buf((size_t)a1nx*a1ny*sizeof(int)*2);

    struct AlignP { int dw,dh, ox0,oy0,onx,ony, px0,py0,pnx,pny, pmin,pmax, prevConst,_pad; };
    struct WP { int num_tx,num_ty, TX0,TY0,TNX,TNY, dwL,dhL; };

    for (int c = 1; c < N; ++c) {
        id<MTLBuffer> craw = C.buf((size_t)w*h*2);
        std::memcpy(craw.contents, frames[c], (size_t)w*h*2);
        if(!down2(craw,w,h,cL0,w0,h0)) return false;
        if(!gdown4(cL0,w0,h0,cL1,w1,h1)) return false;
        if(!gdown4(cL1,w1,h1,cL2,w2,h2)) return false;

        // level 2 (prev const 0). Bind a2 at index 2 as a dummy — prevConst
        // means the kernel never reads it, but Metal requires the slot bound.
        AlignP P2{w2,h2, -1,-1,a2nx,a2ny, 0,0,0,0, 0,0, 1,0};
        if(!run1(C,@"k_align",@[refL2,cL2,a2,a2],{{4,{&P2,sizeof(P2)}}},a2nx,a2ny)) return false;
        // level 1 (prev a2, clamp [-4,3])
        AlignP P1{w1,h1, -1,-1,a1nx,a1ny, -1,-1,a2nx,a2ny, -4,3, 0,0};
        if(!run1(C,@"k_align",@[refL1,cL1,a2,a1],{{4,{&P1,sizeof(P1)}}},a1nx,a1ny)) return false;
        // level 0 (prev a1, clamp [-20,15]) -> write into a0All slot c-1
        id<MTLBuffer> a0slot = C.buf((size_t)num_tx*num_ty*sizeof(int)*2);
        AlignP P0{w0,h0, 0,0,num_tx,num_ty, -1,-1,a1nx,a1ny, -20,15, 0,0};
        if(!run1(C,@"k_align",@[refL0,cL0,a1,a0slot],{{4,{&P0,sizeof(P0)}}},num_tx,num_ty)) return false;
        std::memcpy((uint8_t*)a0All.contents + (size_t)(c-1)*num_tx*num_ty*sizeof(int)*2,
                    a0slot.contents, (size_t)num_tx*num_ty*sizeof(int)*2);

        // weight into wAll slot c-1
        id<MTLBuffer> wslot = C.buf((size_t)TNX*TNY*sizeof(float));
        WP WPp{num_tx,num_ty,TX0,TY0,TNX,TNY,w0,h0};
        if(!run1(C,@"k_weight",@[refL0,cL0,a0slot,wslot],{{4,{&WPp,sizeof(WPp)}}},TNX,TNY)) return false;
        std::memcpy((uint8_t*)wAll.contents + (size_t)(c-1)*TNX*TNY*sizeof(float),
                    wslot.contents, (size_t)TNX*TNY*sizeof(float));
    }

    // total weight
    id<MTLBuffer> total = C.buf((size_t)TNX*TNY*sizeof(float));
    { i2 dim{N-1, TNX*TNY}; if(!run1(C,@"k_total",@[wAll,total],{{2,{&dim,sizeof(dim)}}},TNX*TNY,1)) return false; }

    // spatial merge
    id<MTLBuffer> merged = C.buf((size_t)w*h*2);
    { struct MP { int w,h,num_tx,num_ty,TX0,TY0,TNX,TNY,N,_a,_b; } MPp{w,h,num_tx,num_ty,TX0,TY0,TNX,TNY,N,0,0};
      if(!run1(C,@"k_spatial",@[rawAll,a0All,wAll,total,merged],{{5,{&MPp,sizeof(MPp)}}},w,h)) return false; }

    merged_out.resize((size_t)w*h);
    std::memcpy(merged_out.data(), merged.contents, (size_t)w*h*2);

    // free align/merge buffers before finish (keep `merged`).
    rawAll=nil; a0All=nil; wAll=nil; total=nil; cL0=cL1=cL2=nil; a1=a2=nil; refL0=refL1=refL2=nil;

    // ---------------- finish ----------------
    const uint n = (uint)((size_t)w*h);
    int sox,soy; switch(p.cfa_pattern){
        case CFA_GRBG: sox=1;soy=0;break; case CFA_GBRG: sox=0;soy=1;break;
        case CFA_BGGR: sox=1;soy=1;break; default: sox=0;soy=0;break; }
    id<MTLBuffer> wb=C.buf((size_t)w*h*2);
    { struct FP { int w,h,sox,soy,bp,wp; float wbr,wbg0,wbg1,wbb; } FPp{
        w,h,sox,soy,(int)p.black_point,(int)p.white_point,p.wb_r,p.wb_g0,p.wb_g1,p.wb_b};
      if(!run1(C,@"k_bwl_wb",@[merged,wb],{{2,{&FPp,sizeof(FPp)}}},w,h)) return false; }
    i2 dim{w,h};
    id<MTLBuffer> rgb=C.buf((size_t)w*h*3*2);
    if(!run1(C,@"k_demosaic",@[wb,rgb],{{2,{&dim,sizeof(dim)}}},w,h)) return false;
    wb=nil;
    id<MTLBuffer> srgb=C.buf((size_t)w*h*3*2);
    { id<MTLBuffer> ccm=C.buf(9*sizeof(float)); std::memcpy(ccm.contents,p.ccm,9*sizeof(float));
      if(!run1(C,@"k_srgb",@[rgb,srgb,(id)NSNull.null,ccm],{{2,{&dim,sizeof(dim)}}},(int)n,1)) return false; }
    rgb=nil;
    id<MTLBuffer> gray=C.buf((size_t)w*h*2);
    if(!run1(C,@"k_gray",@[srgb,gray],{{2,{&dim,sizeof(dim)}}},(int)n,1)) return false;

    // ndist LUT
    id<MTLBuffer> ndist=C.buf(65536*sizeof(float));
    { float* nd=(float*)ndist.contents; for(int v=0;v<65536;++v) nd[v]=(float)std::exp(-12.5*std::pow((double)v/65535.0-0.5,2.0)); }

    // tone-map 3-pass exposure fusion
    const double comp=p.compression, gainv=p.gain; const int np=3;
    const double comp_const=1.0+comp/np, gain_const=1.0+gainv/np;
    const double comp_slope=(comp-comp_const)/(np-1), gain_slope=(gainv-gain_const)/(np-1);

    id<MTLBuffer> dark=C.buf((size_t)w*h*2); std::memcpy(dark.contents, gray.contents,(size_t)w*h*2);
    id<MTLBuffer> bx=C.buf((size_t)w*h*sizeof(float));
    id<MTLBuffer> bright=C.buf((size_t)w*h*2), darkg=C.buf((size_t)w*h*2), brightg=C.buf((size_t)w*h*2);
    id<MTLBuffer> bl1=C.buf((size_t)w*h*2), bl2=C.buf((size_t)w*h*2);
    id<MTLBuffer> m1=C.buf((size_t)w*h*sizeof(float)), m2=C.buf((size_t)w*h*sizeof(float));
    id<MTLBuffer> accum=C.buf((size_t)w*h*sizeof(int));
    id<MTLBuffer> comb=C.buf((size_t)w*h*2), tmp=C.buf((size_t)w*h*2);

    auto g7_u16=[&](id<MTLBuffer> in, id<MTLBuffer> out)->bool{
        return run1(C,@"k_g7x_u16",@[in,bx],{{2,{&dim,sizeof(dim)}}},w,h)
            && run1(C,@"k_g7y_u16",@[bx,out],{{2,{&dim,sizeof(dim)}}},w,h); };
    auto g7_f=[&](id<MTLBuffer> in, id<MTLBuffer> out)->bool{
        return run1(C,@"k_g7x_f",@[in,bx],{{2,{&dim,sizeof(dim)}}},w,h)
            && run1(C,@"k_g7y_f",@[bx,out],{{2,{&dim,sizeof(dim)}}},w,h); };

    for (int pass=0; pass<np; ++pass) {
        float norm_comp=(float)(pass*comp_slope+comp_const);
        float norm_gain=(float)(pass*gain_slope+gain_const);
        if(!run1(C,@"k_brighten",@[dark,bright,(id)NSNull.null],{{2,{&norm_comp,sizeof(float)}},{3,{&n,sizeof(uint)}}},(int)n,1)) return false;
        if(!run1(C,@"k_gamma_correct",@[dark,darkg],{{2,{&n,sizeof(uint)}}},(int)n,1)) return false;
        if(!run1(C,@"k_gamma_correct",@[bright,brightg],{{2,{&n,sizeof(uint)}}},(int)n,1)) return false;
        // combine(darkg, brightg) -> comb
        if(!run1(C,@"k_mask_init",@[darkg,brightg,ndist,m1,m2,(id)NSNull.null],{{5,{&n,sizeof(uint)}}},(int)n,1)) return false;
        if(!g7_u16(darkg,bl1)) return false;
        if(!g7_u16(brightg,bl2)) return false;
        if(!run1(C,@"k_band",@[darkg,bl1,brightg,bl2,m1,m2,accum],{{7,{&n,sizeof(uint)}}},(int)n,1)) return false;
        if(!g7_u16(bl1,bl1)) return false;   // bl1 = gauss(bl1)
        if(!g7_u16(bl2,bl2)) return false;
        if(!g7_f(m1,m1)) return false;        // mask = gauss(mask)
        if(!g7_f(m2,m2)) return false;
        if(!run1(C,@"k_residual",@[accum,bl1,m1,bl2,m2,comb],{{6,{&n,sizeof(uint)}}},(int)n,1)) return false;
        // dark = brighten(gamma_inverse(comb), norm_gain)
        if(!run1(C,@"k_gamma_inverse",@[comb,tmp],{{2,{&n,sizeof(uint)}}},(int)n,1)) return false;
        if(!run1(C,@"k_brighten",@[tmp,dark,(id)NSNull.null],{{2,{&norm_gain,sizeof(float)}},{3,{&n,sizeof(uint)}}},(int)n,1)) return false;
    }

    // recolor -> toned (3ch), then gamma_correct (3ch), then contrast+u8
    id<MTLBuffer> toned=C.buf((size_t)w*h*3*2);
    if(!run1(C,@"k_recolor",@[srgb,dark,gray,toned],{{4,{&n,sizeof(uint)}}},(int)n,1)) return false;
    uint n3=n*3;
    if(!run1(C,@"k_gamma_correct",@[toned,toned],{{2,{&n3,sizeof(uint)}}},(int)n3,1)) return false;

    const float strength=5.f; const int black_level=2000;
    const double scale=0.8+0.3/std::min(1.f,strength);
    const double inner=3.141592/(2.0*scale), sinc=std::sin(inner);
    const double slope=65535.0/(2.0*sinc), constant_=slope*sinc;
    const double factor=3.141592/(scale*65535.0), white_scale=65535.0/(65535.0-black_level);
    struct CP { float slope,inner,constant_,factor,white_scale; int black_level; uint n; int _p; }
        CPp{(float)slope,(float)inner,(float)constant_,(float)factor,(float)white_scale,black_level,n3,0};
    id<MTLBuffer> out8=C.buf((size_t)w*h*3);
    if(!run1(C,@"k_contrast_u8",@[toned,out8],{{2,{&CPp,sizeof(CPp)}}},(int)n3,1)) return false;

    rgb_out.resize((size_t)w*h*3);
    std::memcpy(rgb_out.data(), out8.contents, (size_t)w*h*3);
    return true;
    } // autoreleasepool
    return false; // unreachable; satisfies -Werror=return-type
}

}} // namespace hhsr::hdrplus

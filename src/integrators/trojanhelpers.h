NAMESPACE_BEGIN(mitsuba)
#include <drjit/struct.h>

template <typename Float, typename Spectrum> 
struct TrojanContext {
    MI_IMPORT_TYPES();

    Float throughput = 0.f;
    Vector2f s_uv, d_uv;
    Vector3f refract_d = 0.f;

    DRJIT_STRUCT(TrojanContext, throughput, s_uv, d_uv, refract_d)
};

NAMESPACE_END(mitsuba)

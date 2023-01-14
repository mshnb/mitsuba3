NAMESPACE_BEGIN(mitsuba)
#include <drjit/struct.h>

template <typename Float, typename Spectrum> 
struct TrojanContext {
    MI_IMPORT_TYPES();
    //using Vector2f = mitsuba::Vector<Float, 2>;
    //using Vector3f = mitsuba::Vector<Float, 3>;

    //using Mask = dr::mask_t<Float>;

//     TrojanContext(const Vector2f &source = 0, const Vector2f &destination = 0,
//                   const Vector3f &refraction = 0, Float throughput = 0.f)
//         : s_uv(source), d_uv(destination), refract_d(refraction),
//           throughput(throughput) {}

    Float throughput = 0.f;
    Vector2f s_uv, d_uv;
    Vector3f refract_d = 0.f;

    DRJIT_STRUCT(TrojanContext, throughput, s_uv, d_uv, refract_d)
};

NAMESPACE_END(mitsuba)

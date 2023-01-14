#include <random>
#include <drjit/morton.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include "trojanhelpers.h"

NAMESPACE_BEGIN(mitsuba)

/// <summary>
/// trojan means that it may record or compute something interesting while path integrator is working.
/// </summary>
/// <typeparam name="Float"></typeparam>
/// <typeparam name="Spectrum"></typeparam>
template <typename Float, typename Spectrum>
class TrojanPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr)
    using TrojanContext = TrojanContext<Float, Spectrum>;

    TrojanPathIntegrator(const Properties &props) : Base(props) {}

    void render_sample(
        const Scene *scene, Sensor *sensor, Sampler *sampler,
        ImageBlock *block, Float *aovs, const Vector2f &pos, ScalarFloat diff_scale_factor,
                            Mask active = true) const override {
        Film *film     = sensor->film();
        const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);

        ScalarVector2f scale  = 1.f / ScalarVector2f(film->crop_size()),
                       offset = -ScalarVector2f(film->crop_offset()) * scale;

        Vector2f sample_pos   = pos + sampler->next_2d(active),
                 adjusted_pos = dr::fmadd(sample_pos, scale, offset);

        Point2f aperture_sample(.5f);
        if (sensor->needs_aperture_sample())
            aperture_sample = sampler->next_2d(active);

        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0.f)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        Float wavelength_sample = 0.f;
        if constexpr (is_spectral_v<Spectrum>)
            wavelength_sample = sampler->next_1d(active);

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            time, wavelength_sample, adjusted_pos, aperture_sample);

        if (ray.has_differentials)
            ray.scale_differential(diff_scale_factor);

        const Medium *medium = sensor->medium();

        //start our works here
        auto [spec, trojan, valid] = sample_trojan(
            scene, sampler, ray, medium,
                   aovs + (has_alpha ? 5 : 4) /* skip R,G,B,[A],W */, active);

        UnpolarizedSpectrum spec_u = unpolarized_spectrum(ray_weight * spec);

        if (unlikely(has_flag(film->flags(), FilmFlags::Special))) {
            film->prepare_sample(
                spec_u, ray.wavelengths, aovs,
                /*weight*/ 1.f,
                /*alpha */ dr::select(valid, Float(1.f), Float(0.f)), valid);
        } else {
            Color3f rgb;
            if constexpr (is_spectral_v<Spectrum>)
                rgb = spectrum_to_srgb(spec_u, ray.wavelengths, active);
            else if constexpr (is_monochromatic_v<Spectrum>)
                rgb = spec_u.x();
            else
                rgb = spec_u;

            aovs[0] = rgb.x();
            aovs[1] = rgb.y();
            aovs[2] = rgb.z();

            if (likely(has_alpha)) {
                aovs[3] = dr::select(valid, Float(1.f), Float(0.f));
                aovs[4] = 1.f;
            } else {
                aovs[3] = 1.f;
            }
        }

        //rendering result
        block->put(sample_pos, aovs, active);

        //trojan result
        if constexpr (dr::is_jit_v<Float>)
            dr::sync_thread();

        uint32_t spp      = sampler->sample_count();
        size_t size_flat  = spp * dr::prod(film->crop_size());
        size_t size_total = size_flat * sizeof(TrojanContext);
        ScalarFloat *trojan_device =
            (ScalarFloat *) jit_malloc(AllocType::Device, size_total);

        Transform4f fore_object_transform = scene->fore_object_transform();
//         Mask hit_fore                     = trojan.throughput > 0;
//         Vector3f fore_space_d = dr::select(hit_fore, dr::normalize(
//             fore_object_transform.transform_affine(trojan.refract_d)), Vector3f(0.f));
        
        Vector3f fore_space_d = dr::normalize(
                fore_object_transform.transform_affine(trojan.refract_d));
        fore_space_d =
            dr::select(dr::isfinite(fore_space_d), fore_space_d, Vector3f(0.f));

        uint32_t idx = 0;
        dr::store(trojan_device + size_flat * idx++, trojan.throughput);
        dr::store(trojan_device + size_flat * idx++, trojan.s_uv.x());
        dr::store(trojan_device + size_flat * idx++, trojan.s_uv.y());
        dr::store(trojan_device + size_flat * idx++, trojan.d_uv.x());
        dr::store(trojan_device + size_flat * idx++, trojan.d_uv.y());
        dr::store(trojan_device + size_flat * idx++, fore_space_d.x());
        dr::store(trojan_device + size_flat * idx++, fore_space_d.y());
        dr::store(trojan_device + size_flat * idx++, fore_space_d.z());

        ScalarFloat *trojan_host =
            (ScalarFloat *) jit_malloc_migrate(trojan_device, AllocType::Host, 1);

        film->set_trojan_context(trojan_host, size_total);
    }

    std::pair<Spectrum, Bool> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium *medium, Float *aovs,
                                     Bool active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);
        auto [spec, useless, valid] =
            sample_trojan(scene, sampler, ray_, medium, aovs, active);

        return { spec, valid };
    }

    std::tuple<Spectrum, TrojanContext, Bool>
    sample_trojan(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Bool active) const {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        //training: d_uv and refract_d should be the first refraction(after entering fore obj)
        //no training: d_uv and refract_d should be the lastest refraction(leaving fore obj)
        const bool bTrainging = false;

        if (unlikely(m_max_depth == 0))
            return { 0.f, TrojanContext(), false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        UInt32 depth                  = 0;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray                = !m_hide_emitters && dr::neq(scene->environment(), nullptr);


        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        // trojan addition information
        Bool primary_enter_fore      = false;
        Vector3f prev_ray_dir        = 0.f;
        TrojanContext trojan_result  = dr::zeros<TrojanContext>();

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.

           The first argument identifies the loop by name, which is helpful for
           debugging. The subsequent list registers all variables that encode
           the loop state variables. This is crucial: omitting a variable may
           lead to undefined behavior. */
        dr::Loop<Bool> loop("Trojan Path Tracer", sampler, ray, throughput,
                            result, eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, primary_enter_fore,
                            prev_ray_dir, trojan_result, active);

        /* Inform the loop about the maximum number of loop iterations.
           This accelerates wavefront-style rendering by avoiding costly
           synchronization points that check the 'active' flag. */
        loop.set_max_iterations(m_max_depth);

        while (loop(active)) {
            /* dr::Loop implicitly masks all code in the loop using the 'active'
               flag, so there is no need to pass it to every function */

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ dr::eq(depth, 0u));

            // ---------------------- Direct emission ----------------------

            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            if (dr::any_or<true>(dr::neq(si.emitter(scene), nullptr))) {
                DirectionSample3f ds(scene, si, prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                          !prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                result = spec_fma(
                    throughput,
                    ds.emitter->eval(si, prev_bsdf_pdf > 0.f) * mis_bsdf,
                    result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (dr::any_or<true>(active_em)) {
                // Sample the emitter
                auto [ds, em_weight] = scene->sample_emitter_direction(
                    si, sampler->next_2d(), true, active_em);
                active_em &= dr::neq(ds.pdf, 0.f);

                /* Given the detached emitter sample, recompute its contribution
                   with AD to enable light source optimization. */
                if (dr::grad_enabled(si.p)) {
                    ds.d = dr::normalize(ds.p - si.p);
                    Spectrum em_val = scene->eval_emitter_direction(si, ds, active_em);
                    em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                }

                // Evaluate BSDF * cos(theta)
                Vector3f wo = si.to_local(ds.d);
                auto [bsdf_val, bsdf_pdf] =
                    bsdf->eval_pdf(bsdf_ctx, si, wo, active_em);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Compute the MIS weight
                Float mis_em =
                    dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                // Accumulate, being careful with polarization (see spec_fma)
                result[active_em] = spec_fma(
                    throughput, bsdf_val * em_weight * mis_em, result);
            }

            // ---------------------- BSDF sampling ----------------------

            Float sample_1 = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_sample, bsdf_weight] =
                bsdf->sample(bsdf_ctx, si, sample_1, sample_2, active_next);
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ray)) {
                ray = dr::detach<true>(ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo = si.to_local(ray.d);
                auto [bsdf_val, bsdf_pdf] = bsdf->eval_pdf(bsdf_ctx, si, wo, active);
                bsdf_weight[bsdf_pdf > 0.f] = bsdf_val / dr::detach(bsdf_pdf);
            }

            // ------ Update loop variables based on current interaction ------

            // trojan addition
            // TODO:use mask instead of Bool and Mask variables
            // check refraction with foreground's surface
            Bool through_fore_by_refract = active_next &&
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Transmission) &&
                si.hit_foreground(scene->foreground_id());

            // first enter foreground object
            Mask primary_mask = dr::eq(depth, 0u) && through_fore_by_refract;
            if (dr::any_or<true>(primary_mask)) {
                dr::masked(primary_enter_fore, primary_mask) = true;
                trojan_result.s_uv[primary_mask] = si.uv;
            }

            Mask secondary_mask = false; 
            if constexpr (bTrainging) {
                // TODO: DeltaReflection?
                Bool total_interal_reflection =
                    has_flag(bsdf_sample.sampled_type, BSDFFlags::Reflection);
                secondary_mask = dr::eq(depth, 1u) && primary_enter_fore &&
                    (through_fore_by_refract || total_interal_reflection);
            } else
                secondary_mask = dr::eq(depth, 1u) && primary_enter_fore &&
                                 through_fore_by_refract;

            if (dr::any_or<true>(secondary_mask)) {
                // TODO: use some mask or tag to mark TIR?
                // throughput = 1.f shows valid for training
                dr::masked(trojan_result.throughput, secondary_mask) = 1.f;

                trojan_result.d_uv[secondary_mask]      = si.uv;
                trojan_result.refract_d[secondary_mask] = prev_ray_dir;

                // stop recording other data
                if constexpr (bTrainging)
                    dr::masked(primary_enter_fore, secondary_mask) = false;
            }

            if constexpr (!bTrainging) {
                // (after n bounces) final leave fore by refract
                Mask final_mask = (depth > 1u) && primary_enter_fore &&
                                  through_fore_by_refract;
                if (dr::any_or<true>(final_mask)) {
                    // use this info only for rendering instead of training
                    dr::masked(primary_enter_fore, final_mask) = false;

                    trojan_result.d_uv[final_mask]      = si.uv;
                    trojan_result.refract_d[final_mask] = prev_ray_dir;
                }
            }

            throughput *= bsdf_weight;
            eta *= bsdf_sample.eta;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            prev_si = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);
            prev_ray_dir = ray.d;

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            Mask rr_active = depth >= m_rr_depth,
                 rr_continue = sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            active = active_next && (!rr_active || rr_continue) &&
                     dr::neq(throughput_max, 0.f);
        }

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* trojan = */ trojan_result,
            /* valid = */ valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("TrojanPathIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(TrojanPathIntegrator, MonteCarloIntegrator)
MI_EXPORT_PLUGIN(TrojanPathIntegrator, "Trojan Path Tracer integrator");
NAMESPACE_END(mitsuba)

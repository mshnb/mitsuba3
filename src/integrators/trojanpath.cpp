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
    MI_IMPORT_BASE(MonteCarloIntegrator, should_stop, m_max_depth, m_rr_depth,
                   m_tir_depth, m_merge_tir, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr, ShapePtr, Sensor, ImageBlock, Film)
    using TrojanContext = TrojanContext<Float, Spectrum>;

    TrojanPathIntegrator(const Properties &props) : Base(props) {
        int tir_depth = props.get<int>("tir", 0);
        if (tir_depth < 0)
            Throw("\"tir_depth\" must be a value >= 0");

        m_tir_depth = (uint32_t) tir_depth;

        m_merge_tir = props.get<bool>("merge_tir", false);
    }

    void init_trojan(Sensor *sensor, uint32_t spp) override {
        Film *film        = sensor->film();
        if(film->check_trojan_context())
            return;

        size_t size_flat  = spp * dr::prod(film->crop_size());
        size_t size_total = size_flat * sizeof(TrojanContext);

        ScalarFloat *trojan_host =
            (ScalarFloat *) jit_malloc(AllocType::Host, size_total);

        memset(trojan_host, 0, size_total);
        
        film->set_trojan_context(trojan_host, size_total);
    }

    void render_block(const Scene *scene,
                              Sensor *sensor,
                              Sampler *sampler,
                              ImageBlock *block,
                              Float *aovs,
                              uint32_t sample_count,
                              uint32_t seed,
                              uint32_t block_id,
                              uint32_t block_size) const override {
        if constexpr (!dr::is_array_v<Float>) {
            uint32_t pixel_count = block_size * block_size;

            // Avoid overlaps in RNG seeding RNG when a seed is manually specified
            seed += block_id * pixel_count;

            // Scale down ray differentials when tracing multiple rays per pixel
            Float diff_scale_factor = dr::rsqrt((Float) sample_count);

            // Clear block (it's being reused)
            block->clear();

            for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
                sampler->seed(seed + i);

                Point2u pos = dr::morton_decode<Point2u>(i);
                if (dr::any(pos >= block->size()))
                    continue;

                ScalarPoint2f pos_f = ScalarPoint2f(Point2i(pos) + block->offset());
                for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                    render_sample(scene, sensor, sampler, block, aovs,
                                pos_f, diff_scale_factor, j);
                    sampler->advance();
                }
            }
        } else {
            DRJIT_MARK_USED(scene);
            DRJIT_MARK_USED(sensor);
            DRJIT_MARK_USED(sampler);
            DRJIT_MARK_USED(block);
            DRJIT_MARK_USED(aovs);
            DRJIT_MARK_USED(sample_count);
            DRJIT_MARK_USED(seed);
            DRJIT_MARK_USED(block_id);
            DRJIT_MARK_USED(block_size);
            Throw("Not implemented for JIT arrays.");
        }
    }

    void render_sample(
        const Scene *scene, Sensor *sensor, Sampler *sampler,
        ImageBlock *block, Float *aovs, const Vector2f &pos, ScalarFloat diff_scale_factor,
                            uint32_t sample_id, Mask active = true) const override {
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
        uint32_t spp      = sampler->sample_count();
        size_t size_flat  = spp * dr::prod(film->crop_size());
        size_t size_total = size_flat * sizeof(TrojanContext);

        if constexpr (dr::is_jit_v<Float>)
        {
            dr::sync_thread();

            ScalarFloat *trojan_device =
                (ScalarFloat *) jit_malloc(AllocType::Device, size_total);

            Transform4f fore_object_transform = scene->fore_object_transform();
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
        else
        {
            Transform4f fore_object_transform = scene->fore_object_transform();
            Vector3f fore_space_d = dr::normalize(
                    fore_object_transform.transform_affine(trojan.refract_d));
            fore_space_d =
                dr::select(dr::isfinite(fore_space_d), fore_space_d, Vector3f(0.f));

            ScalarVector2u crop_size = film->crop_size();
            ScalarPoint2u pixel_offset = ScalarPoint2u(floor(pos.x()), floor(pos.y()));

            uint32_t pixel_index = pixel_offset.y() * crop_size.x() + pixel_offset.x();
            uint32_t sample_index = pixel_index * spp + sample_id;

            uint32_t idx = 0;
            film->put_trojan_context(&trojan.throughput, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&trojan.s_uv + 0, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&trojan.s_uv + 1, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&trojan.d_uv + 0, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&trojan.d_uv + 1, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&fore_space_d + 0, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&fore_space_d + 1, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
            film->put_trojan_context((ScalarFloat*)&fore_space_d + 2, sizeof(ScalarFloat), (sample_index + size_flat * idx++) * sizeof(ScalarFloat));
        }
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

        if (unlikely(m_max_depth == 0))
            return { 0.f, TrojanContext(), false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f;
        UInt32 depth                  = 0;
        UInt32 tir                    = 0;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray                = !m_hide_emitters && dr::neq(scene->environment(), nullptr);


        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        // trojan addition information
        Bool recording               = false;
        Vector2f prev_src_uv         = 0.f;
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
                            result, eta, depth, tir, valid_ray, prev_si,
                            prev_bsdf_pdf, prev_bsdf_delta, recording,
                            prev_src_uv, prev_ray_dir, trojan_result, active);

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
            Bool active_next = (depth + 1 < m_max_depth) && si.is_valid() && (tir <= m_tir_depth);

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
            throughput *= bsdf_weight;
            eta *= bsdf_sample.eta;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // trojan addition
            Mask hit_fore =
                active_next && si.hit_foreground(scene->foreground_id());

            Mask first_refract = dr::eq(depth, 0u) &&
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Transmission);

            // record start
            Mask enter_fore = hit_fore && first_refract;
            if (dr::any_or<true>(enter_fore)) {
                dr::masked(recording, enter_fore) = true;
            }

            Mask subsequent_tir = hit_fore && depth > 0u &&
                has_flag(bsdf_sample.sampled_type, BSDFFlags::DeltaReflection);
            dr::masked(tir, recording && subsequent_tir) += 1;

            Mask leave_fore  = depth > 0u &&
                has_flag(bsdf_sample.sampled_type, BSDFFlags::Transmission);
            Mask reach_tir      = depth > 0u && (m_tir_depth == 0 || tir > m_tir_depth);
            Mask stop_record    = hit_fore && recording && (leave_fore || reach_tir);
            if (dr::any_or<true>(stop_record)) {
                // record merged data
                Mask record = stop_record;

                //only record target tir bounce
                dr::masked(record, !m_merge_tir) = record && reach_tir;

                //TODO current throughput = 1.f shows valid for training
                dr::masked(trojan_result.throughput, record) = 1.f;

                trojan_result.s_uv[record]         = prev_src_uv;
                trojan_result.refract_d[record]    = prev_ray_dir;
                trojan_result.d_uv[record]         = si.uv;

                // stop recording other data
                dr::masked(recording, stop_record) = false;
            }

            prev_src_uv       = si.uv;
            prev_ray_dir      = ray.d;

            // Information about the current vertex needed by the next iteration
            prev_si = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            //Float rr_prob = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            //Mask rr_active = depth >= m_rr_depth,
            //     rr_continue = sampler->next_1d() < rr_prob;

            ///* Differentiable variants of the renderer require the the russian
            //   roulette sampling weight to be detached to avoid bias. This is a
            //   no-op in non-differentiable variants. */
            //throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            active = active_next && dr::neq(throughput_max, 0.f);
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

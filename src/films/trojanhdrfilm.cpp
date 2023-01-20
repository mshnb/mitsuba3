#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/imageblock.h>

#include <mutex>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class TrojanHDRFilm final : public Film<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset,
                    m_sample_border, m_filter, m_flags)
    MI_IMPORT_TYPES(ImageBlock)

    TrojanHDRFilm(const Properties &props) : Base(props) {
        std::string file_format = string::to_lower(
            props.string("file_format", "openexr"));
        std::string pixel_format = string::to_lower(
            props.string("pixel_format", "rgb"));
        std::string component_format = string::to_lower(
            props.string("component_format", "float16"));

        if (file_format == "openexr" || file_format == "exr")
            m_file_format = Bitmap::FileFormat::OpenEXR;
        else if (file_format == "rgbe")
            m_file_format = Bitmap::FileFormat::RGBE;
        else if (file_format == "pfm")
            m_file_format = Bitmap::FileFormat::PFM;
        else {
            Throw("The \"file_format\" parameter must either be "
                  "equal to \"openexr\", \"pfm\", or \"rgbe\","
                  " found %s instead.", file_format);
        }

        if (pixel_format == "luminance" || is_monochromatic_v<Spectrum>) {
            m_pixel_format = Bitmap::PixelFormat::Y;
            m_flags = +FilmFlags::Empty;
            if (pixel_format != "luminance")
                Log(Warn,
                    "Monochrome mode enabled, setting film output pixel format "
                    "to 'luminance' (was %s).",
                    pixel_format);
        } else if (pixel_format == "luminance_alpha") {
            m_pixel_format = Bitmap::PixelFormat::YA;
            m_flags = +FilmFlags::Alpha;
        } else if (pixel_format == "rgb") {
            m_pixel_format = Bitmap::PixelFormat::RGB;
            m_flags = +FilmFlags::Empty;
        } else if (pixel_format == "rgba") {
            m_pixel_format = Bitmap::PixelFormat::RGBA;
            m_flags = +FilmFlags::Alpha;
        } else if (pixel_format == "xyz") {
            m_pixel_format = Bitmap::PixelFormat::XYZ;
            m_flags = +FilmFlags::Empty;
        } else if (pixel_format == "xyza") {
            m_pixel_format = Bitmap::PixelFormat::XYZA;
            m_flags = +FilmFlags::Alpha;
        } else {
            Throw("The \"pixel_format\" parameter must either be equal to "
                  "\"luminance\", \"luminance_alpha\", \"rgb\", \"rgba\", "
                  " \"xyz\", \"xyza\". Found %s.",
                  pixel_format);
        }

        if (component_format == "float16")
            m_component_format = Struct::Type::Float16;
        else if (component_format == "float32")
            m_component_format = Struct::Type::Float32;
        else if (component_format == "uint32")
            m_component_format = Struct::Type::UInt32;
        else
            Throw("The \"component_format\" parameter must either be "
                  "equal to \"float16\", \"float32\", or \"uint32\"."
                  " Found %s instead.", component_format);

        if (m_file_format == Bitmap::FileFormat::RGBE) {
            if (m_pixel_format != Bitmap::PixelFormat::RGB) {
                Log(Warn, "The RGBE format only supports pixel_format=\"rgb\"."
                           " Overriding..");
                m_pixel_format = Bitmap::PixelFormat::RGB;
            }
            if (m_component_format != Struct::Type::Float32) {
                Log(Warn, "The RGBE format only supports "
                           "component_format=\"float32\". Overriding..");
                m_component_format = Struct::Type::Float32;
            }
        } else if (m_file_format == Bitmap::FileFormat::PFM) {
            // PFM output; override pixel & component format if necessary
            if (m_pixel_format != Bitmap::PixelFormat::RGB && m_pixel_format != Bitmap::PixelFormat::Y) {
                Log(Warn, "The PFM format only supports pixel_format=\"rgb\""
                           " or \"luminance\". Overriding (setting to \"rgb\")..");
                m_pixel_format = Bitmap::PixelFormat::RGB;
            }
            if (m_component_format != Struct::Type::Float32) {
                Log(Warn, "The PFM format only supports"
                           " component_format=\"float32\". Overriding..");
                m_component_format = Struct::Type::Float32;
            }
        }

        //do not need other filter, box is fine
//         m_filter =
//             PluginManager::instance()->create_object<ReconstructionFilter>(
//                 Properties("box"));

        props.mark_queried("banner"); // no banner in Mitsuba 3
    }

    size_t prepare(const std::vector<std::string> &aovs) override {
        bool alpha = has_flag(m_flags, FilmFlags::Alpha);
        size_t base_channels = alpha ? 5 : 4;

        std::vector<std::string> channels(base_channels + aovs.size());

        // Add basic RGBAW channels to the film
        const char *base_channel_names = alpha ? "RGBAW" : "RGBW";

        for (size_t i = 0; i < base_channels; ++i)
            channels[i] = std::string(1, base_channel_names[i]);

        for (size_t i = 0; i < aovs.size(); ++i)
            channels[base_channels + i] = aovs[i];

        /* locked */ {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_storage = new ImageBlock(m_crop_size, m_crop_offset,
                                       (uint32_t) channels.size());
            m_channels = channels;
        }

        std::sort(channels.begin(), channels.end());
        auto it = std::unique(channels.begin(), channels.end());
        if (it != channels.end())
            Throw("Film::prepare(): duplicate channel name \"%s\"", *it);

        return m_channels.size();
    }

    ref<ImageBlock> create_block(const ScalarVector2u &size, bool normalize,
                                 bool border) override {
        bool warn = !dr::is_jit_v<Float> && !is_spectral_v<Spectrum> &&
                    m_channels.size() <= 5;

        bool default_config = size == ScalarVector2u(0);

        return new ImageBlock(default_config ? m_crop_size : size,
                              default_config ? m_crop_offset : ScalarPoint2u(0),
                              (uint32_t) m_channels.size(), m_filter.get(),
                              border /* border */,
                              normalize /* normalize */,
                              dr::is_llvm_v<Float> /* coalesce */,
                              warn /* warn_negative */,
                              warn /* warn_invalid */);
    }
    
    bool check_trojan_context() override {
        return m_trojan_context_ptr != nullptr;
    }

    void set_trojan_context(void *ptr, size_t size) override {
        m_trojan_context_ptr = ptr;
        m_trojan_context_size = size;
    }

    void put_trojan_context(void *ptr, size_t size, uint32_t offset) override {
        Assert(m_trojan_context_ptr != nullptr);

        //std::lock_guard<std::mutex> lock(m_mutex);
        memcpy((char*)m_trojan_context_ptr + offset, ptr, size);
    }

    void put_block(const ImageBlock *block) override {
        Assert(m_storage != nullptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage->put_block(block);
    }

    TensorXf develop(bool raw = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        if (raw) {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_storage->tensor();
        }

        if constexpr (dr::is_jit_v<Float>) {
            Float data;
            uint32_t source_ch;
            uint32_t pixel_count;
            ScalarVector2i size;

            /* locked */ {
                std::lock_guard<std::mutex> lock(m_mutex);
                data        = m_storage->tensor().array();
                size        = m_storage->size();
                source_ch   = (uint32_t) m_storage->channel_count();
                pixel_count = dr::prod(m_storage->size());
            }

            /* The following code develops weighted image block data into
               an output image of the desired configuration, while using
               a minimal number of JIT kernel launches. */

            // Determine what channels are needed
            bool to_xyz    = m_pixel_format == Bitmap::PixelFormat::XYZ ||
                             m_pixel_format == Bitmap::PixelFormat::XYZA;
            bool to_y      = m_pixel_format == Bitmap::PixelFormat::Y ||
                             m_pixel_format == Bitmap::PixelFormat::YA;

            // Number of arbitrary output variables (AOVs)
            bool alpha = has_flag(m_flags, FilmFlags::Alpha);
            uint32_t base_ch = alpha ? 5 : 4,
                     aovs    = source_ch - base_ch;

            /// Number of desired color components
            uint32_t color_ch = to_y ? 1 : 3;

            // Number of channels of the target tensor
            uint32_t target_ch = color_ch + aovs + (uint32_t) alpha;

            // Index vectors referencing pixels & channels of the output image
            UInt32 idx         = dr::arange<UInt32>(pixel_count * target_ch),
                   pixel_idx   = idx / target_ch,
                   channel_idx = dr::fmadd(pixel_idx, uint32_t(-(int) target_ch), idx);

            /* Index vectors referencing source pixels/weights as follows:
                 values_idx = R1, G1, B1, R2, G2, B2 (for RGB output)
                 weight_idx = W1, W1, W1, W2, W2, W2 */
            UInt32 values_idx = dr::fmadd(pixel_idx, source_ch, channel_idx),
                   weight_idx = dr::fmadd(pixel_idx, source_ch, base_ch - 1);

            // If AOVs are desired, their indices in 'values_idx' must be shifted
            if (aovs) {
                // Index of first AOV channel in output image
                uint32_t first_aov = color_ch + (uint32_t) alpha;
                values_idx[channel_idx >= first_aov] += base_ch - first_aov;
            }

            // If luminance + alpha, shift alpha channel to skip the GB channels
            if (alpha && to_y)
                values_idx[dr::eq(channel_idx, color_ch /* alpha */)] += 2;

            Mask value_mask = true;

            // XYZ/Y mode: don't gather color, will be computed below
            if (to_xyz || to_y)
                value_mask = values_idx >= color_ch;

            // Gather the pixel values from the image data buffer
            Float weight = dr::gather<Float>(data, weight_idx),
                  values = dr::gather<Float>(data, values_idx, value_mask);

            // Fill color channels with XYZ/Y data if requested
            if (to_xyz || to_y) {
                UInt32 in_idx  = dr::arange<UInt32>(pixel_count) * source_ch,
                       out_idx = dr::arange<UInt32>(pixel_count) * target_ch;

                Color3f rgb = Color3f(dr::gather<Float>(data, in_idx),
                                      dr::gather<Float>(data, in_idx + 1),
                                      dr::gather<Float>(data, in_idx + 2));

                if (to_y) {
                    dr::scatter(values, luminance(rgb), out_idx);
                } else {
                    Color3f xyz = srgb_to_xyz(rgb);
                    dr::scatter(values, xyz[0], out_idx);
                    dr::scatter(values, xyz[1], out_idx + 1);
                    dr::scatter(values, xyz[2], out_idx + 2);
                }
            }

            // Perform the weight division unless the weight is zero
            values /= dr::select(dr::eq(weight, 0.f), 1.f, weight);

            size_t shape[3] = { (size_t) size.y(), (size_t) size.x(),
                                target_ch };

            return TensorXf(values, 3, shape);
        } else {
            ref<Bitmap> source = bitmap();
            ScalarVector2i size = source->size();
            size_t width = source->channel_count() * dr::prod(size);
            auto data = dr::load<DynamicBuffer<ScalarFloat>>(source->data(), width);

            size_t shape[3] = { (size_t) source->height(),
                                (size_t) source->width(),
                                source->channel_count() };

            return TensorXf(data, 3, shape);
        }
    }

    ref<Bitmap> bitmap(bool raw = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        std::lock_guard<std::mutex> lock(m_mutex);
        auto &&storage = dr::migrate(m_storage->tensor().array(), AllocType::Host);

        if constexpr (dr::is_jit_v<Float>)
            dr::sync_thread();

        bool alpha = has_flag(m_flags, FilmFlags::Alpha);
        uint32_t base_ch = alpha ? 5 : 4;
        bool has_aovs  = m_channels.size() != base_ch;

        Bitmap::PixelFormat source_fmt = !has_aovs
                                     ? (alpha ? Bitmap::PixelFormat::RGBAW
                                              : Bitmap::PixelFormat::RGBW)
                                     : Bitmap::PixelFormat::MultiChannel;

        ref<Bitmap> source = new Bitmap(
            source_fmt, struct_type_v<ScalarFloat>, m_storage->size(),
            m_storage->channel_count(), m_channels, (uint8_t *) storage.data());

        if (raw)
            return source;

        bool to_rgb    = m_pixel_format == Bitmap::PixelFormat::RGB ||
                         m_pixel_format == Bitmap::PixelFormat::RGBA;
        bool to_xyz    = m_pixel_format == Bitmap::PixelFormat::XYZ ||
                         m_pixel_format == Bitmap::PixelFormat::XYZA;
        bool to_y      = m_pixel_format == Bitmap::PixelFormat::Y ||
                         m_pixel_format == Bitmap::PixelFormat::YA;

        uint32_t img_ch = to_y ? 1 : 3;
        uint32_t aovs_channel = has_aovs ? (img_ch + (uint32_t) alpha) : 0;
        uint32_t target_ch =
            (uint32_t) m_storage->channel_count() - base_ch + aovs_channel;

        ref<Bitmap> target = new Bitmap(
            has_aovs ? Bitmap::PixelFormat::MultiChannel : m_pixel_format,
            struct_type_v<ScalarFloat>, m_storage->size(),
            has_aovs ? target_ch : 0);

        if (has_aovs) {
            source->struct_()->operator[](base_ch - 1).flags |=
                +Struct::Flags::Weight;

            for (size_t i = 0; i < target_ch; ++i) {
                Struct::Field &dest_field = target->struct_()->operator[](i);

                switch (i) {
                    case 0:
                        if (to_rgb) {
                            dest_field.name = "R";
                            break;
                        } else if (to_xyz) {
                            dest_field.name = "X";
                            dest_field.blend = {
                                { 0.412453f, "R" },
                                { 0.357580f, "G" },
                                { 0.180423f, "B" }
                            };
                            break;
                        } else if (to_y) {
                            dest_field.name = "Y";
                            dest_field.blend = {
                                { 0.212671f, "R" },
                                { 0.715160f, "G" },
                                { 0.072169f, "B" }
                            };
                            break;
                        }
                        [[fallthrough]];

                    case 1:
                        if (to_rgb) {
                            dest_field.name = "G";
                            break;
                        } else if (to_xyz) {
                            dest_field.name = "Y";
                            dest_field.blend = {
                                { 0.212671f, "R" },
                                { 0.715160f, "G" },
                                { 0.072169f, "B" }
                            };
                            break;
                        } else if (to_y && alpha) {
                            dest_field.name = "A";
                            break;
                        }
                        [[fallthrough]];

                    case 2:
                        if (to_rgb) {
                            dest_field.name = "B";
                            break;
                        } else if (to_xyz) {
                            dest_field.name = "Z";
                            dest_field.blend = {
                                { 0.019334f, "R" },
                                { 0.119193f, "G" },
                                { 0.950227f, "B" }
                            };
                            break;
                        }
                        [[fallthrough]];

                    case 3:
                        if ((to_rgb || to_xyz) && alpha) {
                            dest_field.name = "A";
                            break;
                        }
                        [[fallthrough]];

                    default:
                        dest_field.name = m_channels[base_ch + i - aovs_channel];
                        break;
                }
            }
        }

        source->convert(target);

        return target;
    }

    void write(const fs::path &path) const override {
        fs::path filename = path;
        std::string proper_extension;
        if (m_file_format == Bitmap::FileFormat::OpenEXR)
            proper_extension = ".exr";
        else if (m_file_format == Bitmap::FileFormat::RGBE)
            proper_extension = ".rgbe";
        else
            proper_extension = ".pfm";

        std::string extension = string::to_lower(filename.extension().string());
        if (extension != proper_extension)
            filename.replace_extension(proper_extension);

        #if !defined(_WIN32)
            Log(Info, "\U00002714  Developing \"%s\" ..", filename.string());
        #else
            Log(Info, "Developing \"%s\" ..", filename.string());
        #endif

        ref<Bitmap> source = bitmap();
        if (m_component_format != struct_type_v<ScalarFloat>) {
            // Mismatch between the current format and the one expected by the film
            // Conversion is necessary before saving to disk
            std::vector<std::string> channel_names;
            for (size_t i = 0; i < source->channel_count(); i++)
                channel_names.push_back(source->struct_()->operator[](i).name);
            ref<Bitmap> target = new Bitmap(
                source->pixel_format(),
                m_component_format,
                source->size(),
                source->channel_count(),
                channel_names);
            source->convert(target);

            target->write(filename, m_file_format);
        } else {
            source->write(filename, m_file_format);
        }

        // write trojan_host to file
        if (m_trojan_context_ptr && m_trojan_context_size > 0) {
            filename.replace_extension(".bin");

#if !defined(_WIN32)
            Log(Info, "\U00002714  Writing \"%s\" ..", filename.string());
#else
            Log(Info, "Writing \"%s\" ..", filename.string());
#endif

            ref<FileStream> fs =
                new FileStream(filename, FileStream::ETruncReadWrite);
            fs->write(m_trojan_context_ptr, m_trojan_context_size);

            fs->close();
            jit_free(m_trojan_context_ptr);
        }
    }

    void schedule_storage() override {
        dr::schedule(m_storage->tensor());
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "TrojanHDRFilm[" << std::endl
            << "  size = " << m_size << "," << std::endl
            << "  crop_size = " << m_crop_size << "," << std::endl
            << "  crop_offset = " << m_crop_offset << "," << std::endl
            << "  sample_border = " << m_sample_border << "," << std::endl
            << "  filter = " << m_filter << "," << std::endl
            << "  file_format = " << m_file_format << "," << std::endl
            << "  pixel_format = " << m_pixel_format << "," << std::endl
            << "  component_format = " << m_component_format << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    Bitmap::FileFormat m_file_format;
    Bitmap::PixelFormat m_pixel_format;
    Struct::Type m_component_format;
    ref<ImageBlock> m_storage;
    mutable std::mutex m_mutex;
    std::vector<std::string> m_channels;

    void *m_trojan_context_ptr;
    size_t m_trojan_context_size;
};

MI_IMPLEMENT_CLASS_VARIANT(TrojanHDRFilm, Film)
MI_EXPORT_PLUGIN(TrojanHDRFilm, "Trojan HDR Film")
NAMESPACE_END(mitsuba)

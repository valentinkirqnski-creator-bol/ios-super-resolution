#include "tuning_json.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace hhsr {

// ---- tiny flat-JSON object parser --------------------------------------
// Handles {"key": <value>, ...} where value is a number, true, false, or null.
// String values and nested objects/arrays are skipped (not needed for tuning).
namespace {

struct JsonParser {
    const char* p;
    const char* end;
    explicit JsonParser(const std::string& s) : p(s.c_str()), end(s.c_str() + s.size()) {}

    void skip_ws() {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')) ++p;
    }
    bool parse_string(std::string& out) {
        if (p >= end || *p != '"') return false;
        ++p;
        out.clear();
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {  // minimal escape handling
                ++p;
                switch (*p) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    default: out.push_back(*p); break;
                }
            } else {
                out.push_back(*p);
            }
            ++p;
        }
        if (p < end && *p == '"') { ++p; return true; }
        return false;
    }
    // Skip a value whose type we don't store (string, object, array).
    void skip_value() {
        skip_ws();
        if (p >= end) return;
        if (*p == '"') { std::string tmp; parse_string(tmp); return; }
        if (*p == '{' || *p == '[') {
            const char open = *p, close = (open == '{') ? '}' : ']';
            int depth = 0;
            while (p < end) {
                if (*p == '"') { std::string tmp; parse_string(tmp); continue; }
                if (*p == open) ++depth;
                else if (*p == close) { --depth; if (depth == 0) { ++p; return; } }
                ++p;
            }
            return;
        }
        // scalar token (number / true / false / null): read to delimiter
        while (p < end && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r')
            ++p;
    }
    // Parse a scalar value into `out`; returns false if it was a non-scalar
    // (string/object/array) that we skipped instead.
    bool parse_scalar(double& out) {
        skip_ws();
        if (p >= end) return false;
        if (*p == 't') { out = 1.0; while (p < end && *p != ',' && *p != '}') ++p; return true; }
        if (*p == 'f') { out = 0.0; while (p < end && *p != ',' && *p != '}') ++p; return true; }
        if (*p == 'n') { while (p < end && *p != ',' && *p != '}') ++p; return false; }  // null
        if (*p == '"' || *p == '{' || *p == '[') { skip_value(); return false; }
        char* np = nullptr;
        double v = std::strtod(p, &np);
        if (np == p) { skip_value(); return false; }
        p = np;
        out = v;
        return true;
    }
    // Parse the top-level object into kv. Tolerant: stops at end of input.
    bool parse_object(std::map<std::string, double>& kv) {
        skip_ws();
        if (p >= end || *p != '{') return false;
        ++p;
        while (true) {
            skip_ws();
            if (p >= end) break;
            if (*p == '}') { ++p; break; }
            std::string key;
            if (!parse_string(key)) return false;
            skip_ws();
            if (p >= end || *p != ':') return false;
            ++p;
            double val = 0.0;
            if (parse_scalar(val)) kv[key] = val;  // non-scalars already skipped
        }
        return true;
    }
};

}  // namespace

// ---- key -> Config mapping (mirror of SRBridge.mm ApplyTuningParams) -----
void apply_tuning_kv(Config& cfg, const std::map<std::string, double>& kv) {
    auto has = [&](const char* k) { return kv.find(k) != kv.end(); };
    auto fv  = [&](const char* k) { return (float)kv.at(k); };
    auto iv  = [&](const char* k) { return (int)std::lround(kv.at(k)); };
    auto bv  = [&](const char* k) { return kv.at(k) != 0.0; };

    // --- robustness constants / alignment grey ---
    if (has("r_t"))  cfg.r_t  = fv("r_t");
    if (has("r_s1")) cfg.r_s1 = fv("r_s1");
    if (has("r_s2")) cfg.r_s2 = fv("r_s2");
    if (has("r_Mt")) cfg.r_Mt = fv("r_Mt");
    if (has("alignment_grey_fft"))
        cfg.grey_method = bv("alignment_grey_fft") ? GreyMethod::FFT : GreyMethod::Decimate;
    if (has("decimate_dense_align_enabled")) {
        cfg.decimate_dense_align_enabled = bv("decimate_dense_align_enabled");
        if (cfg.decimate_dense_align_enabled) cfg.grey_method = GreyMethod::Decimate;
    }
    if (has("hf_min_texture_snr")) cfg.hf_min_texture_snr = fv("hf_min_texture_snr");
    if (has("flow_regularize_aperture_ratio"))
        cfg.flow_regularize_aperture_ratio = std::max(0.f, std::min(1.f, fv("flow_regularize_aperture_ratio")));
    if (has("flow_reject_1d_ambiguity_ratio"))
        cfg.flow_reject_1d_ambiguity_ratio = std::max(1.f, fv("flow_reject_1d_ambiguity_ratio"));

    // --- merge quality knobs ---
    if (has("k_detail"))  cfg.k_detail  = fv("k_detail");
    if (has("k_denoise")) cfg.k_denoise = fv("k_denoise");
    if (has("k_stretch")) cfg.k_stretch = fv("k_stretch");
    if (has("k_shrink"))  cfg.k_shrink  = fv("k_shrink");
    if (has("snr_auto_tune")) cfg.snr_auto_tune = bv("snr_auto_tune");

    // --- alignment ---
    if (has("alignment_tile_size")) {
        const int ts = iv("alignment_tile_size");
        cfg.alignment_tile_size = (ts == 8 || ts == 16 || ts == 32 || ts == 64) ? ts : 0;
    }
    if (has("coarsest_search_radius_delta"))
        cfg.coarsest_search_radius_delta = std::max(-5, std::min(5, iv("coarsest_search_radius_delta")));
    if (has("global_prealignment_enabled")) cfg.global_prealignment_enabled = bv("global_prealignment_enabled");
    if (has("global_prealignment_choose_reference"))
        cfg.global_prealignment_choose_reference = bv("global_prealignment_choose_reference");
    if (has("global_prealignment_rotation_range_deg"))
        cfg.global_prealignment_rotation_range_deg = std::max(0.f, std::min(2.f, fv("global_prealignment_rotation_range_deg")));
    if (has("global_prealignment_rotation_step_deg"))
        cfg.global_prealignment_rotation_step_deg = std::max(0.05f, std::min(1.f, fv("global_prealignment_rotation_step_deg")));
    if (has("global_prealignment_max_shift"))
        cfg.global_prealignment_max_shift = std::max(0, std::min(64, iv("global_prealignment_max_shift")));

    // --- robustness / output ---
    if (has("robustness_enabled"))   cfg.robustness_enabled = bv("robustness_enabled");
    if (has("robustness_save_mask")) cfg.robustness_save_mask = bv("robustness_save_mask");
    if (has("structure_map_save")) cfg.structure_map_save = bv("structure_map_save");
    if (has("cref_enabled")) cfg.cref_enabled = bv("cref_enabled");
    if (has("ref_demosaic_fallback")) cfg.ref_demosaic_fallback = bv("ref_demosaic_fallback");
    if (has("hfnn_enabled")) cfg.hfnn_enabled = bv("hfnn_enabled");
    if (has("hfnn_gate")) cfg.hfnn_gate = fv("hfnn_gate");
    if (has("hfnn_strictness")) cfg.hfnn_strictness = fv("hfnn_strictness");
    if (has("mann_enabled")) cfg.mann_enabled = bv("mann_enabled");
    if (has("mann_gate")) cfg.mann_gate = fv("mann_gate");
    if (has("mann_strictness")) cfg.mann_strictness = fv("mann_strictness");
    if (has("fb_conf_enabled")) cfg.fb_conf_enabled = bv("fb_conf_enabled");
    if (has("fb_conf_full")) cfg.fb_conf_full = bv("fb_conf_full");
    if (has("fb_conf_tau_px")) cfg.fb_conf_tau_px = fv("fb_conf_tau_px");
    if (has("fb_conf_subdiv")) cfg.fb_conf_subdiv = std::max(1, std::min(8, iv("fb_conf_subdiv")));
    if (has("fb_consensus_enabled")) cfg.fb_consensus_enabled = bv("fb_consensus_enabled");
    if (has("fb_consensus_tau_px")) cfg.fb_consensus_tau_px = fv("fb_consensus_tau_px");
    if (has("fb_conf_along_edges")) cfg.fb_conf_along_edges = bv("fb_conf_along_edges");
    if (has("ref_demosaic_share_lo")) cfg.ref_demosaic_share_lo = fv("ref_demosaic_share_lo");
    if (has("ref_demosaic_share_hi")) cfg.ref_demosaic_share_hi = fv("ref_demosaic_share_hi");
    if (has("dng_lossless_compress"))
        cfg.dng_codec = bv("dng_lossless_compress") ? Config::DNG_CODEC_LJPEG : Config::DNG_CODEC_NONE;
    if (has("dng_codec")) cfg.dng_codec = std::max(0, std::min(2, iv("dng_codec")));
    if (has("dng_store_unwhitened")) cfg.dng_store_unwhitened = bv("dng_store_unwhitened");
    if (has("merge_arch")) cfg.merge_arch = iv("merge_arch");
    if (has("merge_backend_d3d11")) cfg.merge_backend_d3d11 = bv("merge_backend_d3d11");
    if (has("jpeg_match_python14")) cfg.jpeg_match_python14 = bv("jpeg_match_python14");

    // --- ISP (Config fields; finishing globals g_isp/g_hdr are iOS-only) ---
    if (has("isp_enabled"))        cfg.isp.enabled = bv("isp_enabled");
    if (has("isp_exposure_ev"))    cfg.isp.exposure_ev = fv("isp_exposure_ev");
    if (has("isp_highlight_knee")) cfg.isp.highlight_knee = fv("isp_highlight_knee");
    if (has("isp_local_strength")) cfg.isp.local_strength = fv("isp_local_strength");
    if (has("isp_highlight"))      cfg.isp.highlight_rolloff = fv("isp_highlight");
    if (has("isp_shadow"))         cfg.isp.shadow_lift = fv("isp_shadow");
    if (has("isp_black_point"))    cfg.isp.black_point = fv("isp_black_point");
    if (has("isp_warmth"))         cfg.isp.warmth = fv("isp_warmth");
    if (has("isp_colour_strength")) cfg.isp.colour_strength = fv("isp_colour_strength");
    if (has("isp_contrast"))       cfg.isp.contrast = fv("isp_contrast");
    if (has("isp_vibrance"))       cfg.isp.vibrance = fv("isp_vibrance");
    if (has("isp_chroma_denoise")) cfg.isp.chroma_denoise = fv("isp_chroma_denoise");
    if (has("isp_chroma_radius"))  cfg.isp.chroma_denoise_radius = fv("isp_chroma_radius");
    if (has("isp_saturation"))     cfg.isp.saturation = fv("isp_saturation");
    if (has("isp_local_contrast")) cfg.isp.local_contrast = fv("isp_local_contrast");
    if (has("isp_skin_protect"))   cfg.isp.skin_protect = bv("isp_skin_protect");

    // --- alignment feature flags ---
    if (has("align_match_14"))          cfg.align_match_14 = bv("align_match_14");
    if (has("affine_flow_enabled"))     cfg.affine_flow_enabled = bv("affine_flow_enabled");
    if (has("global_homography_enabled")) cfg.global_homography_enabled = bv("global_homography_enabled");
    if (has("isa_prealign_enabled"))    cfg.isa_prealign_enabled = bv("isa_prealign_enabled");
    if (has("isa_prealign_rot_range_deg")) cfg.isa_prealign_rot_range_deg = fv("isa_prealign_rot_range_deg");
    if (has("isa_prealign_refine_dim")) cfg.isa_prealign_refine_dim = iv("isa_prealign_refine_dim");
    if (has("flow_upsample_candidates")) cfg.flow_upsample_candidates = bv("flow_upsample_candidates");
    if (has("bm_subpixel_enabled"))     cfg.bm_subpixel_enabled = bv("bm_subpixel_enabled");
    if (has("lk_every_level_enabled"))  cfg.lk_every_level_enabled = bv("lk_every_level_enabled");
    if (has("halftile_refine_enabled")) cfg.halftile_refine_enabled = bv("halftile_refine_enabled");
    if (has("halftile_refine_target_ts")) cfg.halftile_refine_target_ts = iv("halftile_refine_target_ts");
    if (has("dense_flow_enabled"))      cfg.dense_flow_enabled = bv("dense_flow_enabled");
    if (has("feat_metric_refine_enabled")) cfg.feat_metric_refine_enabled = bv("feat_metric_refine_enabled");

    // --- guide ---
    if (has("real_rgb_guide"))     cfg.real_rgb_guide = bv("real_rgb_guide");
    if (has("guide_white_balance")) cfg.guide_white_balance = bv("guide_white_balance");
    if (has("guide_color_matrix")) cfg.guide_color_matrix = bv("guide_color_matrix");
    if (has("guide_curve"))        cfg.guide_curve = iv("guide_curve");

    // --- robustness variants ---
    if (has("robustness_per_channel"))    cfg.robustness_per_channel = bv("robustness_per_channel");
    if (has("robustness_perceptual_sqrt")) cfg.robustness_perceptual_sqrt = bv("robustness_perceptual_sqrt");
    if (has("hf_artifact_removal_enabled")) cfg.hf_artifact_removal_enabled = bv("hf_artifact_removal_enabled");
    if (has("hf_variance_loss_threshold")) cfg.hf_variance_loss_threshold = fv("hf_variance_loss_threshold");
    if (has("flow_consensus_enabled"))    cfg.flow_consensus_enabled = bv("flow_consensus_enabled");
    if (has("flow_consensus_reject_px"))  cfg.flow_consensus_reject_px = fv("flow_consensus_reject_px");
    if (has("flow_consensus_search_radius")) cfg.flow_consensus_search_radius = iv("flow_consensus_search_radius");
    if (has("flow_consensus_robust_c"))   cfg.flow_consensus_robust_c = fv("flow_consensus_robust_c");
    if (has("flow_fb_consistency_enabled")) cfg.flow_fb_consistency_enabled = bv("flow_fb_consistency_enabled");
    if (has("flow_fb_alpha"))             cfg.flow_fb_alpha = fv("flow_fb_alpha");
    if (has("flow_fb_beta"))              cfg.flow_fb_beta = fv("flow_fb_beta");
    if (has("motion_geom_reject_enabled")) cfg.motion_geom_reject_enabled = bv("motion_geom_reject_enabled");

    // --- GCam / SABRE ---
    if (has("robustness_gcam_enabled")) cfg.robustness_gcam_enabled = bv("robustness_gcam_enabled");
    if (has("gcam_mult"))      cfg.gcam_mult = fv("gcam_mult");
    if (has("gcam_thresh"))    cfg.gcam_thresh = fv("gcam_thresh");
    if (has("gcam_boost"))     cfg.gcam_boost = fv("gcam_boost");
    if (has("gcam_tex_gate"))  cfg.gcam_tex_gate = fv("gcam_tex_gate");
    if (has("gcam_irr_gate"))  cfg.gcam_irr_gate = fv("gcam_irr_gate");
    if (has("gcam_green_k"))   cfg.gcam_green_k = fv("gcam_green_k");
    if (has("gcam_dilate_soft")) cfg.gcam_dilate_soft = fv("gcam_dilate_soft");
    if (has("gcam_filter_var_scale")) cfg.gcam_filter_var_scale = fv("gcam_filter_var_scale");
    if (has("gcam_sabre_green_factor")) cfg.gcam_sabre_green_factor = fv("gcam_sabre_green_factor");
    if (has("gcam_sabre_clip_point"))   cfg.gcam_sabre_clip_point = fv("gcam_sabre_clip_point");
    if (has("gcam_sabre_k_rgb"))        cfg.gcam_sabre_k_rgb = fv("gcam_sabre_k_rgb");
    if (has("gcam_sabre_k_green"))      cfg.gcam_sabre_k_green = fv("gcam_sabre_k_green");
    if (has("gcam_sabre_rob_noise"))    cfg.gcam_sabre_rob_noise = fv("gcam_sabre_rob_noise");
    if (has("gcam_sabre_mult_rgb"))     cfg.gcam_sabre_mult_rgb = fv("gcam_sabre_mult_rgb");
    if (has("gcam_sabre_mult_green"))   cfg.gcam_sabre_mult_green = fv("gcam_sabre_mult_green");
    if (has("gcam_sabre_boost"))        cfg.gcam_sabre_boost = fv("gcam_sabre_boost");
    if (has("gcam_sabre_boost_var_mult")) cfg.gcam_sabre_boost_var_mult = fv("gcam_sabre_boost_var_mult");
    if (has("gcam_sabre_motion_thresh")) cfg.gcam_sabre_motion_thresh = fv("gcam_sabre_motion_thresh");
    if (has("gcam_sabre_dilate_deadzone")) cfg.gcam_sabre_dilate_deadzone = fv("gcam_sabre_dilate_deadzone");
    if (has("gcam_sabre_dilate_div"))   cfg.gcam_sabre_dilate_div = fv("gcam_sabre_dilate_div");
    if (has("gcam_sabre_gpu"))          cfg.gcam_sabre_gpu = bv("gcam_sabre_gpu");
    if (has("gcam_sabre_gpu_guide"))    cfg.gcam_sabre_gpu_guide = bv("gcam_sabre_gpu_guide");
    if (has("gcam_sabre_raw_res"))      cfg.gcam_sabre_raw_res = bv("gcam_sabre_raw_res");
    if (has("gcam_sabre_bilinear_upscale")) cfg.gcam_sabre_bilinear_upscale = bv("gcam_sabre_bilinear_upscale");
    if (has("gcam_sabre_min5x5"))       cfg.gcam_sabre_min5x5 = bv("gcam_sabre_min5x5");
    if (has("gcam_sabre_merge_res"))    cfg.gcam_sabre_merge_res = bv("gcam_sabre_merge_res");
    if (has("gcam_sabre_min_window"))   cfg.gcam_sabre_min_window = std::max(1, std::min(31, iv("gcam_sabre_min_window")));
    if (has("gcam_sabre_noise_model"))  cfg.gcam_sabre_noise_model = bv("gcam_sabre_noise_model");
    if (has("gcam_sabre_dilate_2x"))    cfg.gcam_sabre_dilate_2x = bv("gcam_sabre_dilate_2x");
    if (has("gcam_sabre_rejection_only")) cfg.gcam_sabre_rejection_only = bv("gcam_sabre_rejection_only");
    if (has("gcam_sabre_flow_bilinear")) cfg.gcam_sabre_flow_bilinear = bv("gcam_sabre_flow_bilinear");
    if (has("gcam_structure_match")) cfg.gcam_structure_match = bv("gcam_structure_match");
    if (has("gcam_unblocker_enabled")) cfg.gcam_unblocker_enabled = bv("gcam_unblocker_enabled");
    if (has("gcam_unblocker_scale")) cfg.gcam_unblocker_scale = fv("gcam_unblocker_scale");
    if (has("gcam_unblocker_offset")) cfg.gcam_unblocker_offset = fv("gcam_unblocker_offset");
    if (has("gcam_sabre_kernel"))       cfg.gcam_sabre_kernel = bv("gcam_sabre_kernel");
    if (has("gcam_cov_shrinked_sigma")) cfg.gcam_cov_shrinked_sigma = fv("gcam_cov_shrinked_sigma");
    if (has("gcam_cov_stretched_sigma")) cfg.gcam_cov_stretched_sigma = fv("gcam_cov_stretched_sigma");
    if (has("gcam_cov_gradient_clip"))  cfg.gcam_cov_gradient_clip = fv("gcam_cov_gradient_clip");
    if (has("gcam_cov_base_sigma"))     cfg.gcam_cov_base_sigma = fv("gcam_cov_base_sigma");
    if (has("gcam_cov_blurred_sigma"))  cfg.gcam_cov_blurred_sigma = fv("gcam_cov_blurred_sigma");
    if (has("gcam_cov_gradient_transition")) cfg.gcam_cov_gradient_transition = fv("gcam_cov_gradient_transition");
    if (has("gcam_cov_scale"))          cfg.gcam_cov_scale = fv("gcam_cov_scale");
    if (has("gcam_sabre_k_rgb_biquad")) cfg.gcam_sabre_k_rgb_biquad = fv("gcam_sabre_k_rgb_biquad");
    if (has("gcam_sabre_k_green_biquad")) cfg.gcam_sabre_k_green_biquad = fv("gcam_sabre_k_green_biquad");

    // --- mask fetch / geom reject / edge misalign ---
    if (has("robustness_guide_res_bilinear")) cfg.robustness_guide_res_bilinear = bv("robustness_guide_res_bilinear");
    if (has("merge_robustness_bilinear")) cfg.merge_robustness_bilinear = bv("merge_robustness_bilinear");
    if (has("robustness_bicubic_guide")) cfg.robustness_bicubic_guide = bv("robustness_bicubic_guide");
    if (has("motion_geom_reject_threshold")) cfg.motion_geom_reject_threshold = fv("motion_geom_reject_threshold");
    if (has("motion_geom_relative")) cfg.motion_geom_relative = bv("motion_geom_relative");
    if (has("motion_geom_noise_floor_mult")) cfg.motion_geom_noise_floor_mult = fv("motion_geom_noise_floor_mult");
    if (has("motion_geom_reject_threshold_relative")) cfg.motion_geom_reject_threshold_relative = fv("motion_geom_reject_threshold_relative");
    if (has("edge_misalign_enabled"))  cfg.edge_misalign_enabled = bv("edge_misalign_enabled");
    if (has("edge_misalign_edge_snr")) cfg.edge_misalign_edge_snr = fv("edge_misalign_edge_snr");
    if (has("edge_misalign_radius"))   cfg.edge_misalign_radius = iv("edge_misalign_radius");
    if (has("edge_misalign_shift_z"))  cfg.edge_misalign_shift_z = fv("edge_misalign_shift_z");
    if (has("edge_misalign_ghost_z"))  cfg.edge_misalign_ghost_z = fv("edge_misalign_ghost_z");
    if (has("edge_misalign_min_conf")) cfg.edge_misalign_min_conf = fv("edge_misalign_min_conf");
    if (has("align_ambiguous_fallback_enabled")) cfg.align_ambiguous_fallback_enabled = bv("align_ambiguous_fallback_enabled");
    if (has("frame_discard_enabled"))  cfg.frame_discard_enabled = bv("frame_discard_enabled");
    if (has("frame_discard_min_fraction")) cfg.frame_discard_min_fraction = fv("frame_discard_min_fraction");
    if (has("debug_noise_model_disabled")) cfg.debug_noise_model_disabled = bv("debug_noise_model_disabled");
    if (has("kernel_selection_linear"))
        cfg.selection = bv("kernel_selection_linear") ? SelectionLaw::Linear : SelectionLaw::HardThreshold;
    if (has("robustness_raw_resolution_enabled")) cfg.robustness_raw_resolution_enabled = bv("robustness_raw_resolution_enabled");

    // --- learned robustness nets ---
    if (has("use_neural_robustness"))  cfg.use_neural_robustness = bv("use_neural_robustness");
    if (has("robustness_refine_nn_enabled")) cfg.robustness_refine_nn_enabled = bv("robustness_refine_nn_enabled");
    if (has("rob_mask_nn_enabled"))    cfg.rob_mask_nn_enabled = bv("rob_mask_nn_enabled");
    if (has("robust_nn_enabled"))      cfg.robust_nn_enabled = bv("robust_nn_enabled");
    if (has("exposure_match_enabled")) cfg.exposure_match_enabled = bv("exposure_match_enabled");
    if (has("robust_nn_v15_enabled"))  cfg.robust_nn_v15_enabled = bv("robust_nn_v15_enabled");
    if (has("robust_nn_reject_below")) cfg.robust_nn_reject_below = fv("robust_nn_reject_below");
    if (has("robust_nn_accept_above")) cfg.robust_nn_accept_above = fv("robust_nn_accept_above");
    if (has("robustness_refine_max_reduction")) cfg.robustness_refine_max_reduction = fv("robustness_refine_max_reduction");
    if (has("robustness_refine_deadzone")) cfg.robustness_refine_deadzone = fv("robustness_refine_deadzone");
}

bool apply_tuning_json_file(Config& cfg, const std::string& path, std::string* err) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { if (err) *err = "cannot open " + path; return false; }
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    std::map<std::string, double> kv;
    JsonParser jp(text);
    if (!jp.parse_object(kv)) { if (err) *err = "not a JSON object: " + path; return false; }
    apply_tuning_kv(cfg, kv);
    return true;
}

}  // namespace hhsr

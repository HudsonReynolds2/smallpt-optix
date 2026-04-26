// Must be defined before any include that transitively pulls in <cmath>.
#ifdef _WIN32
#  ifndef _USE_MATH_DEFINES
#    define _USE_MATH_DEFINES
#  endif
#endif

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "shared.h"
#include "scene.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// PHASE 3 CHANGES vs PHASE 2:
//   1. Spheres tessellated into the same triangle GAS as the walls.
//      No more sphere GAS, no more IAS, no more built-in sphere IS module.
//      pipelineCompileOptions.usesPrimitiveTypeFlags = TRIANGLE only.
//      traversableGraphFlags = ALLOW_SINGLE_GAS.
//   2. Tile-based launch loop. Single full-image accum buffer; each tile
//      launch is (TILE_W, TILE_H, 1) with tile_origin_{x,y} in Params.
//      The raygen offsets its launch index by the tile origin.
//      This bounds OptiX per-launch state regardless of full image size.
// =============================================================================

// Tile dimensions for tile-based launches.
// 512x512 was the plan's suggestion. Set to 0 to disable tiling (single
// full-image launch) for A/B comparison: `cmake -DTILE_SIZE=0 ...`
#ifndef TILE_W
#define TILE_W 512
#endif
#ifndef TILE_H
#define TILE_H 512
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t rc = (call);                                                \
        if (rc != cudaSuccess) {                                                \
            std::cerr << "CUDA error " << cudaGetErrorString(rc)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while(0)

#define OPTIX_CHECK(call)                                                       \
    do {                                                                        \
        OptixResult rc = (call);                                                \
        if (rc != OPTIX_SUCCESS) {                                              \
            std::cerr << "OptiX error " << optixGetErrorName(rc)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while(0)

#define OPTIX_CHECK_LOG(call)                                                   \
    do {                                                                        \
        char   log[4096]; size_t log_size = sizeof(log);                       \
        OptixResult rc = (call);                                                \
        if (log_size > 1) std::cerr << "OptiX log: " << log << "\n";           \
        if (rc != OPTIX_SUCCESS) {                                              \
            std::cerr << "OptiX error " << optixGetErrorName(rc)               \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";        \
            std::exit(1);                                                       \
        }                                                                       \
    } while(0)

template<typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    if (level <= 2)
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

static std::string load_ptx(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PTX file: " + path);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static void write_ppm(const char* path, unsigned int w, unsigned int h, const float4* buf) {
    FILE* fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Cannot open output file: %s\n", path); return; }
    fprintf(fp, "P3\n%u %u\n255\n", w, h);
    for (unsigned int i = 0; i < w * h; ++i) {
        auto to_byte = [](float v) -> int {
            v = powf(fmaxf(fminf(v, 1.0f), 0.0f), 1.0f/2.2f);
            return (int)(v * 255.0f + 0.5f);
        };
        fprintf(fp, "%d %d %d ", to_byte(buf[i].x), to_byte(buf[i].y), to_byte(buf[i].z));
    }
    fclose(fp);
}

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s [--width W] [--height H] [--spp N] [--output FILE] [--ptx FILE]\n", prog);
    fprintf(stderr, "  Defaults: 1024x768, 256 spp, output.ppm, shaders.ptx\n");
}

int main(int argc, char* argv[]) {
    unsigned int width   = 1024;
    unsigned int height  = 768;
    unsigned int spp     = 256;
    const char*  outfile = "output.ppm";
    std::string  ptx_path = "shaders.ptx";

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i],"--width")  && i+1<argc) width   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--height") && i+1<argc) height  = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--spp")    && i+1<argc) spp     = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--output") && i+1<argc) outfile = argv[++i];
        else if (!strcmp(argv[i],"--ptx")    && i+1<argc) ptx_path= argv[++i];
        else if (!strcmp(argv[i],"--help"))  { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown arg: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    try {
        // ---------------------------------------------------------------
        // Init CUDA + OptiX context
        // ---------------------------------------------------------------
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());

        OptixDeviceContext context = nullptr;
        {
            OptixDeviceContextOptions opts = {};
            opts.logCallbackFunction = &context_log_cb;
            opts.logCallbackLevel    = 4;
#ifndef NDEBUG
            opts.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
            OPTIX_CHECK(optixDeviceContextCreate(0, &opts, &context));
        }

        // ---------------------------------------------------------------
        // Build the unified triangle GAS (walls + tessellated spheres).
        // PHASE 3: spheres are now triangles in this same GAS. No second
        // GAS, no IAS.
        // ---------------------------------------------------------------
        std::vector<float3>   scene_verts;
        std::vector<uint32_t> tri_sbt_idx;
        build_scene_geometry(WALL_TESS_N, SPHERE_TESS_LAT, SPHERE_TESS_LON,
                             scene_verts, tri_sbt_idx);
        const size_t num_vertices  = scene_verts.size();
        const size_t num_triangles = num_vertices / 3;

        std::cerr << "Scene geometry: walls "
                  << NUM_WALLS << " * " << triangles_per_wall(WALL_TESS_N)
                  << " tris, spheres "
                  << NUM_SPHERES << " * " << triangles_per_sphere(SPHERE_TESS_LAT, SPHERE_TESS_LON)
                  << " tris, total " << num_triangles << " tris ("
                  << num_vertices << " verts)\n";

        CUdeviceptr d_verts = 0, d_sbt_idx = 0;
        CUDA_CHECK(cudaMalloc((void**)&d_verts, num_vertices * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy((void*)d_verts, scene_verts.data(),
                              num_vertices * sizeof(float3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc((void**)&d_sbt_idx, num_triangles * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy((void*)d_sbt_idx, tri_sbt_idx.data(),
                              num_triangles * sizeof(uint32_t), cudaMemcpyHostToDevice));

        OptixTraversableHandle gas_handle = 0;
        CUdeviceptr            d_gas_output = 0;
        {
            OptixBuildInput build_input = {};
            build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            build_input.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
            build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
            build_input.triangleArray.numVertices         = (unsigned int)num_vertices;
            build_input.triangleArray.vertexBuffers       = &d_verts;

            // One flag per SBT record: NUM_HG_RECORDS records.
            std::vector<unsigned int> flags(NUM_HG_RECORDS, OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
            build_input.triangleArray.flags        = flags.data();
            build_input.triangleArray.numSbtRecords = NUM_HG_RECORDS;

            build_input.triangleArray.sbtIndexOffsetBuffer        = d_sbt_idx;
            build_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
            build_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

            OptixAccelBuildOptions accel_opts = {};
            accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                  | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                                  | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_opts,
                        &build_input, 1, &sizes));

            CUdeviceptr d_temp = 0, d_output_uncompacted = 0, d_compacted_size = 0;
            CUDA_CHECK(cudaMalloc((void**)&d_temp, sizes.tempSizeInBytes));
            CUDA_CHECK(cudaMalloc((void**)&d_output_uncompacted, sizes.outputSizeInBytes));
            CUDA_CHECK(cudaMalloc((void**)&d_compacted_size, sizeof(size_t)));

            OptixAccelEmitDesc emit = {};
            emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit.result = d_compacted_size;

            OPTIX_CHECK(optixAccelBuild(context, 0, &accel_opts, &build_input, 1,
                d_temp, sizes.tempSizeInBytes,
                d_output_uncompacted, sizes.outputSizeInBytes,
                &gas_handle, &emit, 1));

            size_t compacted_size = 0;
            CUDA_CHECK(cudaMemcpy(&compacted_size, (void*)d_compacted_size,
                                  sizeof(size_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMalloc((void**)&d_gas_output, compacted_size));
            OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output,
                                          compacted_size, &gas_handle));

            CUDA_CHECK(cudaFree((void*)d_temp));
            CUDA_CHECK(cudaFree((void*)d_output_uncompacted));
            CUDA_CHECK(cudaFree((void*)d_compacted_size));
        }

        // ---------------------------------------------------------------
        // Module + pipeline (TRIANGLES only; no sphere primitives).
        // ---------------------------------------------------------------
        // Shared log buffer used by all OPTIX_CHECK_LOG calls below.
        char   optix_log[4096];
        size_t optix_log_size = sizeof(optix_log);

        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        {
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

            pipelineCompileOptions.usesMotionBlur        = false;
            // PHASE 3: single GAS, no IAS.
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.numPayloadValues      = 2;
            pipelineCompileOptions.numAttributeValues    = 2;
            pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

            std::string ptx = load_ptx(ptx_path);
            optix_log_size = sizeof(optix_log);
            OptixResult rc = optixModuleCreate(context,
                &moduleCompileOptions, &pipelineCompileOptions,
                ptx.c_str(), ptx.size(), optix_log, &optix_log_size, &module);
            if (optix_log_size > 1) std::cerr << "OptiX log: " << optix_log << "\n";
            if (rc != OPTIX_SUCCESS) {
                std::cerr << "OptiX error " << optixGetErrorName(rc) << " at " << __FILE__ << ":" << __LINE__ << "\n";
                std::exit(1);
            }
        }

        // Program groups: raygen, miss, single triangle hit group.
        OptixProgramGroup rg_pg = nullptr, ms_pg = nullptr, ch_pg = nullptr;
        {
            OptixProgramGroupOptions pg_opts = {};

            OptixProgramGroupDesc rg_desc = {};
            rg_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            rg_desc.raygen.module            = module;
            rg_desc.raygen.entryFunctionName = "__raygen__rg";
            optix_log_size = sizeof(optix_log);
            OPTIX_CHECK(optixProgramGroupCreate(context, &rg_desc, 1, &pg_opts,
                            optix_log, &optix_log_size, &rg_pg));

            OptixProgramGroupDesc ms_desc = {};
            ms_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            ms_desc.miss.module            = module;
            ms_desc.miss.entryFunctionName = "__miss__ms";
            optix_log_size = sizeof(optix_log);
            OPTIX_CHECK(optixProgramGroupCreate(context, &ms_desc, 1, &pg_opts,
                            optix_log, &optix_log_size, &ms_pg));

            OptixProgramGroupDesc ch_desc = {};
            ch_desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            ch_desc.hitgroup.moduleCH             = module;
            ch_desc.hitgroup.entryFunctionNameCH  = "__closesthit__ch";
            optix_log_size = sizeof(optix_log);
            OPTIX_CHECK(optixProgramGroupCreate(context, &ch_desc, 1, &pg_opts,
                            optix_log, &optix_log_size, &ch_pg));
        }

        // Pipeline
        OptixPipeline pipeline = nullptr;
        {
            OptixProgramGroup pgs[] = { rg_pg, ms_pg, ch_pg };
            OptixPipelineLinkOptions linkOptions = {};
            linkOptions.maxTraceDepth = 20; // must match bounce loop in __raygen__rg
            optix_log_size = sizeof(optix_log);
            OPTIX_CHECK(optixPipelineCreate(context,
                &pipelineCompileOptions, &linkOptions, pgs, 3,
                optix_log, &optix_log_size, &pipeline));

            OptixStackSizes stack_sizes = {};
            for (auto& pg : pgs) {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline));
            }
            unsigned int trav_css, state_css, cont_css;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, 20, 0, 0,
                &trav_css, &state_css, &cont_css));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                trav_css, state_css, cont_css, 1)); // single GAS = depth 1
        }

        // ---------------------------------------------------------------
        // SBT: NUM_WALLS records + NUM_SPHERES records, all using ch_pg.
        // ---------------------------------------------------------------
        OptixShaderBindingTable sbt = {};
        CUdeviceptr d_rg_record = 0, d_ms_record = 0, d_hg_records = 0;
        {
            RayGenSbtRecord rg_rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(rg_pg, &rg_rec));
            CUDA_CHECK(cudaMalloc((void**)&d_rg_record, sizeof(RayGenSbtRecord)));
            CUDA_CHECK(cudaMemcpy((void*)d_rg_record, &rg_rec, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

            MissSbtRecord ms_rec;
            ms_rec.data.bg_color = {0.0f, 0.0f, 0.0f};
            OPTIX_CHECK(optixSbtRecordPackHeader(ms_pg, &ms_rec));
            CUDA_CHECK(cudaMalloc((void**)&d_ms_record, sizeof(MissSbtRecord)));
            CUDA_CHECK(cudaMemcpy((void*)d_ms_record, &ms_rec, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

            std::vector<HitGroupSbtRecord> hg_recs(NUM_HG_RECORDS);

            for (int w = 0; w < NUM_WALLS; ++w) {
                OPTIX_CHECK(optixSbtRecordPackHeader(ch_pg, &hg_recs[w]));
                hg_recs[w].data.emission = g_walls[w].emission;
                hg_recs[w].data.albedo   = g_walls[w].albedo;
                hg_recs[w].data.material = g_walls[w].material;
            }
            for (int s = 0; s < NUM_SPHERES; ++s) {
                int idx = NUM_WALLS + s;
                OPTIX_CHECK(optixSbtRecordPackHeader(ch_pg, &hg_recs[idx]));
                hg_recs[idx].data.emission = g_spheres[s].emission;
                hg_recs[idx].data.albedo   = g_spheres[s].albedo;
                hg_recs[idx].data.material = g_spheres[s].material;
            }

            CUDA_CHECK(cudaMalloc((void**)&d_hg_records, NUM_HG_RECORDS * sizeof(HitGroupSbtRecord)));
            CUDA_CHECK(cudaMemcpy((void*)d_hg_records, hg_recs.data(),
                NUM_HG_RECORDS * sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice));

            sbt.raygenRecord                = d_rg_record;
            sbt.missRecordBase              = d_ms_record;
            sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
            sbt.missRecordCount             = 1;
            sbt.hitgroupRecordBase          = d_hg_records;
            sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
            sbt.hitgroupRecordCount         = NUM_HG_RECORDS;
        }

        // ---------------------------------------------------------------
        // Accumulation buffer (full image, allocated ONCE).
        // ---------------------------------------------------------------
        CUdeviceptr d_accum = 0;
        CUDA_CHECK(cudaMalloc((void**)&d_accum, width * height * sizeof(float4)));
        CUDA_CHECK(cudaMemset((void*)d_accum, 0, width * height * sizeof(float4)));

        Params h_params = {};
        h_params.accum_buffer = reinterpret_cast<float4*>(d_accum);
        h_params.width        = width;
        h_params.height       = height;
        h_params.handle       = gas_handle;

        h_params.eye = {50.0f, 52.0f, 295.6f};
        float fov = 0.5135f;
        h_params.cx = {width * fov / height, 0.0f, 0.0f};
        float3 gaze = {0.0f, -0.042612f, -1.0f};
        float glen = sqrtf(gaze.x*gaze.x + gaze.y*gaze.y + gaze.z*gaze.z);
        gaze = {gaze.x/glen, gaze.y/glen, gaze.z/glen};
        float3 cx = h_params.cx;
        float3 cy_dir = {
            cx.y*gaze.z - cx.z*gaze.y,
            cx.z*gaze.x - cx.x*gaze.z,
            cx.x*gaze.y - cx.y*gaze.x
        };
        float cy_len = sqrtf(cy_dir.x*cy_dir.x + cy_dir.y*cy_dir.y + cy_dir.z*cy_dir.z);
        h_params.cy = {cy_dir.x/cy_len * fov, cy_dir.y/cy_len * fov, cy_dir.z/cy_len * fov};

        CUdeviceptr d_params;
        CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(Params)));

        CUstream stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // ---------------------------------------------------------------
        // Tile launch loop.
        // Use TILE_W / TILE_H if both > 0; else single-launch mode for A/B.
        // Time the entire tile loop together (one cudaEvent pair).
        // ---------------------------------------------------------------
        const unsigned int tile_w = (TILE_W > 0) ? (unsigned int)TILE_W : width;
        const unsigned int tile_h = (TILE_H > 0) ? (unsigned int)TILE_H : height;
        const unsigned int n_tiles_x = (width  + tile_w - 1) / tile_w;
        const unsigned int n_tiles_y = (height + tile_h - 1) / tile_h;

        std::cerr << "Launch: tile " << tile_w << "x" << tile_h
                  << ", grid " << n_tiles_x << "x" << n_tiles_y
                  << " = " << (n_tiles_x * n_tiles_y) << " tiles\n";

        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        h_params.samples_per_launch = spp;
        h_params.subframe_index     = 0;

        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        for (unsigned int ty = 0; ty < n_tiles_y; ++ty) {
            for (unsigned int tx = 0; tx < n_tiles_x; ++tx) {
                h_params.tile_origin_x = tx * tile_w;
                h_params.tile_origin_y = ty * tile_h;

                // Tile size for this launch (last col/row may be smaller).
                unsigned int this_w = std::min(tile_w, width  - h_params.tile_origin_x);
                unsigned int this_h = std::min(tile_h, height - h_params.tile_origin_y);

                CUDA_CHECK(cudaMemcpyAsync((void*)d_params, &h_params, sizeof(Params),
                                           cudaMemcpyHostToDevice, stream));
                OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params),
                                        &sbt, this_w, this_h, 1));
            }
        }
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        double primary_rays = (double)width * height * spp;
        double mrays_sec    = primary_rays / (ms * 1e3);

        printf("Resolution: %ux%u  SPP: %u  Time: %.2f ms  Mrays/s: %.2f\n",
               width, height, spp, ms, mrays_sec);
        printf("CSV: optix_phase3,%ux%u,%u,%.2f,%.2f,sm_86\n",
               width, height, spp, ms, mrays_sec);

        // ---------------------------------------------------------------
        // Read back and write PPM
        // ---------------------------------------------------------------
        std::vector<float4> h_accum(width * height);
        CUDA_CHECK(cudaMemcpy(h_accum.data(), (void*)d_accum, width*height*sizeof(float4), cudaMemcpyDeviceToHost));
        write_ppm(outfile, width, height, h_accum.data());
        printf("Wrote: %s\n", outfile);

        // ---------------------------------------------------------------
        // Cleanup
        // ---------------------------------------------------------------
        CUDA_CHECK(cudaFree((void*)d_accum));
        CUDA_CHECK(cudaFree((void*)d_params));
        CUDA_CHECK(cudaFree((void*)d_rg_record));
        CUDA_CHECK(cudaFree((void*)d_ms_record));
        CUDA_CHECK(cudaFree((void*)d_hg_records));
        CUDA_CHECK(cudaFree((void*)d_gas_output));
        CUDA_CHECK(cudaFree((void*)d_verts));
        CUDA_CHECK(cudaFree((void*)d_sbt_idx));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaStreamDestroy(stream));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(ch_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(ms_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(rg_pg));
        OPTIX_CHECK(optixModuleDestroy(module));
        OPTIX_CHECK(optixDeviceContextDestroy(context));

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
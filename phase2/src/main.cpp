#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "shared.h"
#include "scene.h"   // wall + sphere geometry, NUM_WALLS / NUM_SPHERES / NUM_HG_RECORDS, build_wall_geometry()

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

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// SBT record template
// ---------------------------------------------------------------------------
template<typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

// ---------------------------------------------------------------------------
// Logging callback
// ---------------------------------------------------------------------------
static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
    if (level <= 2)
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

// ---------------------------------------------------------------------------
// Load PTX from file
// ---------------------------------------------------------------------------
static std::string load_ptx(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open PTX file: " + path);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------
// Write PPM (gamma corrected)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
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
        // -----------------------------------------------------------------------
        // Init CUDA + OptiX context
        // -----------------------------------------------------------------------
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

        // =======================================================================
        // Build triangle GAS for the 6 walls (tessellated).
        // Each wall is split into a grid of small triangles to avoid float32
        // precision loss in the closest-hit race against the protruding light
        // sphere (see scene_default.h for full discussion). Triangles share
        // SBT records by wall: NUM_WALLS hit group records cover the whole mesh.
        // =======================================================================
        OptixTraversableHandle tri_gas_handle = 0;
        CUdeviceptr            d_tri_gas_output = 0;
        {
            // Tessellate every wall into a grid of small triangles. See
            // scene_default.h for the why -- short version: a primary ray
            // grazing the ceiling near the protruding light sphere needs a
            // closest-hit comparison whose float32 t-error is much smaller
            // than the sphere's 0.27-unit protrusion. A single 100x170 unit
            // ceiling triangle lost that race; ~6-unit cells win it cleanly.
            //
            // build_wall_geometry returns:
            //   verts:        flat vertex array (3 per triangle, no index buffer)
            //   tri_wall_idx: parallel array, one entry per triangle giving
            //                 its wall index in 0..NUM_WALLS-1. We feed this
            //                 directly to OptiX as the SBT index offset, so
            //                 every triangle of wall N points at hit group
            //                 record N (one record per wall, not per triangle).
            std::vector<float3>   verts;
            std::vector<uint32_t> tri_wall_idx;
            build_wall_geometry(WALL_TESS_N, verts, tri_wall_idx);
            const uint32_t num_triangles = (uint32_t)tri_wall_idx.size();

            CUdeviceptr d_verts;
            CUDA_CHECK(cudaMalloc((void**)&d_verts, verts.size() * sizeof(float3)));
            CUDA_CHECK(cudaMemcpy((void*)d_verts, verts.data(),
                                  verts.size() * sizeof(float3), cudaMemcpyHostToDevice));

            CUdeviceptr d_tri_sbt_indices;
            CUDA_CHECK(cudaMalloc((void**)&d_tri_sbt_indices, num_triangles * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy((void*)d_tri_sbt_indices, tri_wall_idx.data(),
                                  num_triangles * sizeof(uint32_t), cudaMemcpyHostToDevice));

            // One flag entry per SBT record (= per wall).
            std::vector<uint32_t> tri_flags(NUM_WALLS, OPTIX_GEOMETRY_FLAG_NONE);

            OptixBuildInput tri_input = {};
            tri_input.type                            = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            tri_input.triangleArray.vertexFormat      = OPTIX_VERTEX_FORMAT_FLOAT3;
            tri_input.triangleArray.numVertices       = (uint32_t)verts.size();
            tri_input.triangleArray.vertexBuffers     = &d_verts;
            tri_input.triangleArray.flags             = tri_flags.data();
            tri_input.triangleArray.numSbtRecords     = NUM_WALLS;
            tri_input.triangleArray.sbtIndexOffsetBuffer        = d_tri_sbt_indices;
            tri_input.triangleArray.sbtIndexOffsetSizeInBytes   = sizeof(uint32_t);
            tri_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

            printf("Wall tessellation: %d walls, %u triangles total (NxN=%d, %d tris/wall)\n",
                   NUM_WALLS, num_triangles, WALL_TESS_N, triangles_per_wall(WALL_TESS_N));

            OptixAccelBuildOptions accel_opts = {};
            accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                  | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_opts, &tri_input, 1, &sizes));

            CUdeviceptr d_temp;
            CUDA_CHECK(cudaMalloc((void**)&d_temp, sizes.tempSizeInBytes));

            size_t compact_offset = (sizes.outputSizeInBytes + 7) & ~7ull;
            CUdeviceptr d_output_and_compact;
            CUDA_CHECK(cudaMalloc((void**)&d_output_and_compact, compact_offset + 8));

            OptixAccelEmitDesc emit = {};
            emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit.result = d_output_and_compact + compact_offset;

            OPTIX_CHECK(optixAccelBuild(context, 0, &accel_opts, &tri_input, 1,
                d_temp, sizes.tempSizeInBytes,
                d_output_and_compact, sizes.outputSizeInBytes,
                &tri_gas_handle, &emit, 1));

            CUDA_CHECK(cudaFree((void*)d_temp));
            CUDA_CHECK(cudaFree((void*)d_tri_sbt_indices));
            CUDA_CHECK(cudaFree((void*)d_verts));

            size_t compact_size;
            CUDA_CHECK(cudaMemcpy(&compact_size, (void*)emit.result, sizeof(size_t), cudaMemcpyDeviceToHost));
            if (compact_size < sizes.outputSizeInBytes) {
                CUDA_CHECK(cudaMalloc((void**)&d_tri_gas_output, compact_size));
                OPTIX_CHECK(optixAccelCompact(context, 0, tri_gas_handle, d_tri_gas_output, compact_size, &tri_gas_handle));
                CUDA_CHECK(cudaFree((void*)d_output_and_compact));
            } else {
                d_tri_gas_output = d_output_and_compact;
            }
        }

        // =======================================================================
        // Build sphere GAS for mirror / glass / light.
        // =======================================================================
        OptixTraversableHandle sph_gas_handle = 0;
        CUdeviceptr            d_sph_gas_output = 0;
        {
            std::vector<float3> centers(NUM_SPHERES);
            std::vector<float>  radii(NUM_SPHERES);
            for (int i = 0; i < NUM_SPHERES; ++i) {
                centers[i] = g_spheres[i].center;
                radii[i]   = g_spheres[i].radius;
            }

            CUdeviceptr d_centers, d_radii;
            CUDA_CHECK(cudaMalloc((void**)&d_centers, NUM_SPHERES * sizeof(float3)));
            CUDA_CHECK(cudaMalloc((void**)&d_radii,   NUM_SPHERES * sizeof(float)));
            CUDA_CHECK(cudaMemcpy((void*)d_centers, centers.data(),
                                  NUM_SPHERES * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy((void*)d_radii,   radii.data(),
                                  NUM_SPHERES * sizeof(float),  cudaMemcpyHostToDevice));

            std::vector<uint32_t> sph_sbt_indices(NUM_SPHERES);
            for (int i = 0; i < NUM_SPHERES; ++i) sph_sbt_indices[i] = i;
            CUdeviceptr d_sph_sbt_indices;
            CUDA_CHECK(cudaMalloc((void**)&d_sph_sbt_indices, NUM_SPHERES * sizeof(uint32_t)));
            CUDA_CHECK(cudaMemcpy((void*)d_sph_sbt_indices, sph_sbt_indices.data(),
                                  NUM_SPHERES * sizeof(uint32_t), cudaMemcpyHostToDevice));

            std::vector<uint32_t> sph_flags(NUM_SPHERES, OPTIX_GEOMETRY_FLAG_NONE);

            OptixBuildInput sph_input = {};
            sph_input.type                                       = OPTIX_BUILD_INPUT_TYPE_SPHERES;
            sph_input.sphereArray.vertexBuffers                  = &d_centers;
            sph_input.sphereArray.numVertices                    = NUM_SPHERES;
            sph_input.sphereArray.radiusBuffers                  = &d_radii;
            sph_input.sphereArray.sbtIndexOffsetBuffer           = d_sph_sbt_indices;
            sph_input.sphereArray.sbtIndexOffsetSizeInBytes      = sizeof(uint32_t);
            sph_input.sphereArray.sbtIndexOffsetStrideInBytes    = sizeof(uint32_t);
            sph_input.sphereArray.flags                          = sph_flags.data();
            sph_input.sphereArray.numSbtRecords                  = NUM_SPHERES;

            OptixAccelBuildOptions accel_opts = {};
            accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                                  | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
            accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_opts, &sph_input, 1, &sizes));

            CUdeviceptr d_temp;
            CUDA_CHECK(cudaMalloc((void**)&d_temp, sizes.tempSizeInBytes));

            size_t compact_offset = (sizes.outputSizeInBytes + 7) & ~7ull;
            CUdeviceptr d_output_and_compact;
            CUDA_CHECK(cudaMalloc((void**)&d_output_and_compact, compact_offset + 8));

            OptixAccelEmitDesc emit = {};
            emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            emit.result = d_output_and_compact + compact_offset;

            OPTIX_CHECK(optixAccelBuild(context, 0, &accel_opts, &sph_input, 1,
                d_temp, sizes.tempSizeInBytes,
                d_output_and_compact, sizes.outputSizeInBytes,
                &sph_gas_handle, &emit, 1));

            CUDA_CHECK(cudaFree((void*)d_temp));
            CUDA_CHECK(cudaFree((void*)d_sph_sbt_indices));
            CUDA_CHECK(cudaFree((void*)d_centers));
            CUDA_CHECK(cudaFree((void*)d_radii));

            size_t compact_size;
            CUDA_CHECK(cudaMemcpy(&compact_size, (void*)emit.result, sizeof(size_t), cudaMemcpyDeviceToHost));
            if (compact_size < sizes.outputSizeInBytes) {
                CUDA_CHECK(cudaMalloc((void**)&d_sph_gas_output, compact_size));
                OPTIX_CHECK(optixAccelCompact(context, 0, sph_gas_handle, d_sph_gas_output, compact_size, &sph_gas_handle));
                CUDA_CHECK(cudaFree((void*)d_output_and_compact));
            } else {
                d_sph_gas_output = d_output_and_compact;
            }
        }

        // =======================================================================
        // Build IAS containing two instances: triangle GAS + sphere GAS.
        // sbtOffset puts walls in records 0..NUM_WALLS-1 and spheres after.
        // =======================================================================
        OptixTraversableHandle ias_handle = 0;
        CUdeviceptr            d_ias_output = 0;
        CUdeviceptr            d_instances  = 0;
        {
            // Identity 3x4 transform (row-major)
            const float identity[12] = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 1.0f, 0.0f,
            };

            OptixInstance instances[2] = {};

            // Triangle instance: SBT records 0..NUM_WALLS-1 (one per wall)
            std::memcpy(instances[0].transform, identity, sizeof(identity));
            instances[0].instanceId        = 0;
            instances[0].sbtOffset         = 0;
            instances[0].visibilityMask    = 1;
            instances[0].flags             = OPTIX_INSTANCE_FLAG_NONE;
            instances[0].traversableHandle = tri_gas_handle;

            // Sphere instance: SBT records starting at NUM_WALLS
            std::memcpy(instances[1].transform, identity, sizeof(identity));
            instances[1].instanceId        = 1;
            instances[1].sbtOffset         = NUM_WALLS;
            instances[1].visibilityMask    = 1;
            instances[1].flags             = OPTIX_INSTANCE_FLAG_NONE;
            instances[1].traversableHandle = sph_gas_handle;

            CUDA_CHECK(cudaMalloc((void**)&d_instances, sizeof(instances)));
            CUDA_CHECK(cudaMemcpy((void*)d_instances, instances, sizeof(instances), cudaMemcpyHostToDevice));

            OptixBuildInput ias_input = {};
            ias_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            ias_input.instanceArray.instances    = d_instances;
            ias_input.instanceArray.numInstances = 2;

            OptixAccelBuildOptions accel_opts = {};
            accel_opts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
            accel_opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

            OptixAccelBufferSizes sizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_opts, &ias_input, 1, &sizes));

            CUdeviceptr d_temp;
            CUDA_CHECK(cudaMalloc((void**)&d_temp, sizes.tempSizeInBytes));
            CUDA_CHECK(cudaMalloc((void**)&d_ias_output, sizes.outputSizeInBytes));

            OPTIX_CHECK(optixAccelBuild(context, 0, &accel_opts, &ias_input, 1,
                d_temp, sizes.tempSizeInBytes,
                d_ias_output, sizes.outputSizeInBytes,
                &ias_handle, nullptr, 0));

            CUDA_CHECK(cudaFree((void*)d_temp));
        }

        // -----------------------------------------------------------------------
        // Load PTX and create modules
        // -----------------------------------------------------------------------
        std::string ptx = load_ptx(ptx_path);

        OptixModule module = nullptr;
        OptixModule sphere_module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_opts = {};
        {
            OptixModuleCompileOptions module_compile_opts = {};
#ifdef NDEBUG
            module_compile_opts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            module_compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
            module_compile_opts.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

            pipeline_compile_opts.usesMotionBlur                   = false;
            // We now use an IAS over two GASes -> single-level instancing
            pipeline_compile_opts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pipeline_compile_opts.numPayloadValues                 = 2;  // lo + hi PRD pointer
            pipeline_compile_opts.numAttributeValues               = 2;  // built-in sphere uses 1, but 2 is safe min for triangles
            pipeline_compile_opts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
            pipeline_compile_opts.pipelineLaunchParamsVariableName = "params";
            pipeline_compile_opts.usesPrimitiveTypeFlags           =
                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

            char log[4096]; size_t log_size = sizeof(log);
            OPTIX_CHECK_LOG(optixModuleCreate(context,
                &module_compile_opts, &pipeline_compile_opts,
                ptx.c_str(), ptx.size(),
                log, &log_size, &module));

            // Built-in sphere intersection module (only used by sphere hit group)
            OptixBuiltinISOptions bis_opts = {};
            bis_opts.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
            bis_opts.usesMotionBlur      = false;
            OPTIX_CHECK(optixBuiltinISModuleGet(context,
                &module_compile_opts, &pipeline_compile_opts,
                &bis_opts, &sphere_module));
        }

        // -----------------------------------------------------------------------
        // Create program groups: raygen, miss, triangle hit group, sphere hit group
        // -----------------------------------------------------------------------
        OptixProgramGroup rg_pg = nullptr, ms_pg = nullptr;
        OptixProgramGroup tri_ch_pg = nullptr, sph_ch_pg = nullptr;
        {
            OptixProgramGroupOptions pg_opts = {};
            char log[4096]; unsigned int log_size;

            OptixProgramGroupDesc rg_desc = {};
            rg_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            rg_desc.raygen.module            = module;
            rg_desc.raygen.entryFunctionName = "__raygen__rg";
            log_size = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &rg_desc, 1, &pg_opts, log, &log_size, &rg_pg));

            OptixProgramGroupDesc ms_desc = {};
            ms_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
            ms_desc.miss.module            = module;
            ms_desc.miss.entryFunctionName = "__miss__ms";
            log_size = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &ms_desc, 1, &pg_opts, log, &log_size, &ms_pg));

            // Triangle hit group: closest-hit only, no IS (built-in triangles use hardware IS)
            OptixProgramGroupDesc tri_ch_desc = {};
            tri_ch_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            tri_ch_desc.hitgroup.moduleCH            = module;
            tri_ch_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            // No moduleIS / entryFunctionNameIS for triangles
            log_size = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &tri_ch_desc, 1, &pg_opts, log, &log_size, &tri_ch_pg));

            // Sphere hit group: closest-hit + built-in sphere IS
            OptixProgramGroupDesc sph_ch_desc = {};
            sph_ch_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            sph_ch_desc.hitgroup.moduleCH            = module;
            sph_ch_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
            sph_ch_desc.hitgroup.moduleIS            = sphere_module;
            sph_ch_desc.hitgroup.entryFunctionNameIS = nullptr;
            log_size = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context, &sph_ch_desc, 1, &pg_opts, log, &log_size, &sph_ch_pg));
        }

        // -----------------------------------------------------------------------
        // Link pipeline
        // -----------------------------------------------------------------------
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t max_depth = 20;
            OptixProgramGroup pgs[] = { rg_pg, ms_pg, tri_ch_pg, sph_ch_pg };

            OptixPipelineLinkOptions link_opts = {};
            link_opts.maxTraceDepth = max_depth;

            char log[4096]; size_t log_size = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(context,
                &pipeline_compile_opts, &link_opts,
                pgs, 4, log, &log_size, &pipeline));

            OptixStackSizes ss = {};
            for (auto pg : pgs)
                OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &ss, pipeline));

            uint32_t trav_css, state_css, cont_css;
            // maxTraversableGraphDepth = 2 (IAS -> GAS)
            OPTIX_CHECK(optixUtilComputeStackSizes(&ss, max_depth, 0, 0,
                &trav_css, &state_css, &cont_css));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
                trav_css, state_css, cont_css, 2));
        }

        // -----------------------------------------------------------------------
        // Build SBT.
        // Layout: [wall records 0..NUM_WALLS-1] [sphere records 0..NUM_SPHERES-1]
        // The triangle instance has sbtOffset=0; sphere instance has sbtOffset=NUM_WALLS.
        // Every triangle of wall N points at hit group record N via the
        // sbtIndexOffsetBuffer set up during GAS build.
        // -----------------------------------------------------------------------
        OptixShaderBindingTable sbt = {};
        CUdeviceptr d_rg_record, d_ms_record, d_hg_records;
        {
            // Raygen
            RayGenSbtRecord rg_rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(rg_pg, &rg_rec));
            CUDA_CHECK(cudaMalloc((void**)&d_rg_record, sizeof(RayGenSbtRecord)));
            CUDA_CHECK(cudaMemcpy((void*)d_rg_record, &rg_rec, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));

            // Miss
            MissSbtRecord ms_rec;
            ms_rec.data.bg_color = {0.0f, 0.0f, 0.0f};
            OPTIX_CHECK(optixSbtRecordPackHeader(ms_pg, &ms_rec));
            CUDA_CHECK(cudaMalloc((void**)&d_ms_record, sizeof(MissSbtRecord)));
            CUDA_CHECK(cudaMemcpy((void*)d_ms_record, &ms_rec, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));

            // Hit groups: one record per wall (shared by all that wall's
            // tessellated triangles), then one per sphere.
            std::vector<HitGroupSbtRecord> hg_recs(NUM_HG_RECORDS);

            for (int w = 0; w < NUM_WALLS; ++w) {
                OPTIX_CHECK(optixSbtRecordPackHeader(tri_ch_pg, &hg_recs[w]));
                hg_recs[w].data.emission = g_walls[w].emission;
                hg_recs[w].data.albedo   = g_walls[w].albedo;
                hg_recs[w].data.material = g_walls[w].material;
            }

            for (int s = 0; s < NUM_SPHERES; ++s) {
                int idx = NUM_WALLS + s;
                OPTIX_CHECK(optixSbtRecordPackHeader(sph_ch_pg, &hg_recs[idx]));
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

        // -----------------------------------------------------------------------
        // Allocate accumulation buffer
        // -----------------------------------------------------------------------
        CUdeviceptr d_accum;
        CUDA_CHECK(cudaMalloc((void**)&d_accum, width * height * sizeof(float4)));
        CUDA_CHECK(cudaMemset((void*)d_accum, 0, width * height * sizeof(float4)));

        // -----------------------------------------------------------------------
        // Set up camera params (smallpt Cornell box camera)
        // -----------------------------------------------------------------------
        Params h_params = {};
        h_params.accum_buffer = reinterpret_cast<float4*>(d_accum);
        h_params.width        = width;
        h_params.height       = height;
        h_params.handle       = ias_handle; // top-level traversable is the IAS now

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

        // -----------------------------------------------------------------------
        // Launch
        // -----------------------------------------------------------------------
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));

        h_params.samples_per_launch = spp;
        h_params.subframe_index     = 0;
        CUDA_CHECK(cudaMemcpy((void*)d_params, &h_params, sizeof(Params), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_params, sizeof(Params), &sbt, width, height, 1));
        CUDA_CHECK(cudaEventRecord(ev_stop, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        double primary_rays = (double)width * height * spp;
        double mrays_sec    = primary_rays / (ms * 1e3);

        printf("Resolution: %ux%u  SPP: %u  Time: %.2f ms  Mrays/s: %.2f\n",
               width, height, spp, ms, mrays_sec);
        printf("CSV: optix_spheres,%ux%u,%u,%.2f,%.2f,sm_86\n",
               width, height, spp, ms, mrays_sec);

        // -----------------------------------------------------------------------
        // Read back and write PPM
        // -----------------------------------------------------------------------
        std::vector<float4> h_accum(width * height);
        CUDA_CHECK(cudaMemcpy(h_accum.data(), (void*)d_accum, width*height*sizeof(float4), cudaMemcpyDeviceToHost));
        write_ppm(outfile, width, height, h_accum.data());
        printf("Wrote: %s\n", outfile);

        // -----------------------------------------------------------------------
        // Cleanup
        // -----------------------------------------------------------------------
        CUDA_CHECK(cudaFree((void*)d_accum));
        CUDA_CHECK(cudaFree((void*)d_params));
        CUDA_CHECK(cudaFree((void*)d_rg_record));
        CUDA_CHECK(cudaFree((void*)d_ms_record));
        CUDA_CHECK(cudaFree((void*)d_hg_records));
        CUDA_CHECK(cudaFree((void*)d_tri_gas_output));
        CUDA_CHECK(cudaFree((void*)d_sph_gas_output));
        CUDA_CHECK(cudaFree((void*)d_ias_output));
        CUDA_CHECK(cudaFree((void*)d_instances));
        CUDA_CHECK(cudaEventDestroy(ev_start));
        CUDA_CHECK(cudaEventDestroy(ev_stop));
        CUDA_CHECK(cudaStreamDestroy(stream));

        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        OPTIX_CHECK(optixProgramGroupDestroy(sph_ch_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(tri_ch_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(ms_pg));
        OPTIX_CHECK(optixProgramGroupDestroy(rg_pg));
        OPTIX_CHECK(optixModuleDestroy(sphere_module));
        OPTIX_CHECK(optixModuleDestroy(module));
        OPTIX_CHECK(optixDeviceContextDestroy(context));

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

#include "imageio.hpp"
#include "sampling.cuh"
#include "specular.cuh"
#include "sphere.hpp"
#include "cuda_tools.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define REFRACTIVE_INDEX_OUT 1.0f
#define REFRACTIVE_INDEX_IN  1.5f

namespace smallpt {

    const Sphere g_spheres[] = {
        Sphere(1e5f,  Vector3(1e5f + 1.0f, 40.8f, 81.6f),    Vector3(),       Vector3(0.75f, 0.25f, 0.25f), Reflection_t::Diffuse),    //Left
        Sphere(1e5f,  Vector3(-1e5f + 99.0f, 40.8f, 81.6f),  Vector3(),       Vector3(0.25f, 0.25f, 0.75f), Reflection_t::Diffuse),    //Right
        Sphere(1e5f,  Vector3(50.0f, 40.8f, 1e5f),           Vector3(),       Vector3(0.75f),               Reflection_t::Diffuse),    //Back
        Sphere(1e5f,  Vector3(50.0f, 40.8f, -1e5f + 170.0f), Vector3(),       Vector3(),                    Reflection_t::Diffuse),    //Front
        Sphere(1e5f,  Vector3(50.0f, 1e5f, 81.6f),           Vector3(),       Vector3(0.75f),               Reflection_t::Diffuse),    //Bottom
        Sphere(1e5f,  Vector3(50.0f, -1e5f + 81.6f, 81.6f),  Vector3(),       Vector3(0.75f),               Reflection_t::Diffuse),    //Top
        Sphere(16.5f, Vector3(27.0f, 16.5f, 47.0f),          Vector3(),       Vector3(0.999f),              Reflection_t::Specular),   //Mirror
        Sphere(16.5f, Vector3(73.0f, 16.5f, 78.0f),          Vector3(),       Vector3(0.999f),              Reflection_t::Refractive), //Glass
        Sphere(600.0f,Vector3(50.0f, 681.6f - 0.27f, 81.6f), Vector3(12.0f),  Vector3(),                    Reflection_t::Diffuse)     //Light
    };

    // Returns the unnormalized outward normal in `n_unnorm` for the closest hit
    // (overwritten as closer hits are found, matching how `ray.m_tmax` is tracked).
    __device__ inline bool Intersect(const Sphere* dev_spheres,
                                     std::size_t nb_spheres,
                                     const Ray& ray,
                                     size_t& id,
                                     Vector3& n_unnorm) {
        bool hit = false;
        Vector3 n_tmp;
        for (std::size_t i = 0u; i < nb_spheres; ++i) {
            if (dev_spheres[i].Intersect(ray, n_tmp)) {
                hit = true;
                id  = i;
                n_unnorm = n_tmp;
            }
        }
        return hit;
    }

    __device__ static Vector3 Radiance(const Sphere* dev_spheres,
                                       std::size_t nb_spheres,
                                       const Ray& ray,
                                       curandState* state) {
        Ray r = ray;
        Vector3 L;
        Vector3 F(1.0f);

        while (true) {
            std::size_t id;
            Vector3 n_unnorm;
            if (!Intersect(dev_spheres, nb_spheres, r, id, n_unnorm)) {
                return L;
            }

            const Sphere& shape = dev_spheres[id];
            const Vector3 p = r(r.m_tmax);
            // Fix C: normal comes from Intersect, which built it from
            // bounded-magnitude quantities (sqrtD * d - perp). The old code
            // was `Normalize(p - shape.m_p)` which subtracts two ~1e5
            // vectors and loses precision for the wall spheres.
            const Vector3 n = Normalize(n_unnorm);

            L += F * shape.m_e;
            F *= shape.m_f;

            // Russian roulette
            if (4 < r.m_depth) {
                const float continue_probability = shape.m_f.Max();
                if (curand_uniform(state) >= continue_probability) {
                    return L;
                }
                F /= continue_probability;
            }

            switch (shape.m_reflection_t) {

            case Reflection_t::Specular: {
                const Vector3 d = IdealSpecularReflect(r.m_d, n);
                r = Ray(p, d, EPSILON_SPHERE, std::numeric_limits<float>::infinity(), r.m_depth + 1u);
                break;
            }

            case Reflection_t::Refractive: {
                float pr;
                const Vector3 d = IdealSpecularTransmit(r.m_d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, pr, state);
                F *= pr;
                r = Ray(p, d, EPSILON_SPHERE, std::numeric_limits<float>::infinity(), r.m_depth + 1u);
                break;
            }

            default: {
                const Vector3 w = (0.0f > n.Dot(r.m_d)) ? n : -n;
                const Vector3 u = Normalize((fabsf(w.m_x) > 0.1f ? Vector3(0.0f, 1.0f, 0.0f) : Vector3(1.0f, 0.0f, 0.0f)).Cross(w));
                const Vector3 v = w.Cross(u);

                const Vector3 sample_d = CosineWeightedSampleOnHemisphere(curand_uniform(state), curand_uniform(state));
                const Vector3 d = Normalize(sample_d.m_x * u + sample_d.m_y * v + sample_d.m_z * w);
                r = Ray(p, d, EPSILON_SPHERE, std::numeric_limits<float>::infinity(), r.m_depth + 1u);
            }
            }
        }
    }

    __global__ static void kernel(const Sphere* dev_spheres,
                                  std::size_t nb_spheres,
                                  std::uint32_t w,
                                  std::uint32_t h,
                                  Vector3* Ls,
                                  std::uint32_t nb_samples) {

        const std::uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
        const std::uint32_t offset = x + y * blockDim.x * gridDim.x;

        if (x >= w || y >= h) {
            return;
        }

        curandState state;
        curand_init(offset, 0u, 0u, &state);

        const Vector3 eye  = { 50.0f, 52.0f, 295.6f };
        const Vector3 gaze = Normalize(Vector3(0.0f, -0.042612f, -1.0f));
        const float fov    = 0.5135f;
        const Vector3 cx   = { w * fov / h, 0.0f, 0.0f };
        const Vector3 cy   = Normalize(cx.Cross(gaze)) * fov;

        for (std::size_t sy = 0u, i = (h - 1u - y) * w + x; sy < 2u; ++sy) {
            for (std::size_t sx = 0u; sx < 2u; ++sx) {
                Vector3 L;
                for (std::size_t s = 0u; s < nb_samples; ++s) {
                    const float u1 = 2.0f * curand_uniform(&state);
                    const float u2 = 2.0f * curand_uniform(&state);
                    const float dx = (u1 < 1.0f) ? sqrtf(u1) - 1.0f : 1.0f - sqrtf(2.0f - u1);
                    const float dy = (u2 < 1.0f) ? sqrtf(u2) - 1.0f : 1.0f - sqrtf(2.0f - u2);
                    const Vector3 d = cx * (((sx + 0.5f + dx) * 0.5f + x) / w - 0.5f) +
                                      cy * (((sy + 0.5f + dy) * 0.5f + y) / h - 0.5f) + gaze;

                    L += Radiance(dev_spheres, nb_spheres,
                        Ray(eye + d * 130.0f, Normalize(d), EPSILON_SPHERE), &state)
                        * (1.0f / nb_samples);
                }
                Ls[i] += 0.25f * Clamp(L);
            }
        }
    }

    static void Render(std::uint32_t w, std::uint32_t h, std::uint32_t nb_samples,
                       const char* output_path) noexcept {

        const std::uint32_t nb_pixels  = w * h;
        const std::size_t   nb_spheres = sizeof(g_spheres) / sizeof(g_spheres[0]);

        Sphere* dev_spheres;
        HANDLE_ERROR(cudaMalloc((void**)&dev_spheres, sizeof(g_spheres)));
        HANDLE_ERROR(cudaMemcpy(dev_spheres, g_spheres, sizeof(g_spheres), cudaMemcpyHostToDevice));

        Vector3* dev_Ls;
        HANDLE_ERROR(cudaMalloc((void**)&dev_Ls, nb_pixels * sizeof(Vector3)));
        HANDLE_ERROR(cudaMemset(dev_Ls, 0, nb_pixels * sizeof(Vector3)));

        // Pad grid up to next multiple of 16 so bounds check in kernel handles odd resolutions
        const dim3 nthreads(16u, 16u);
        const dim3 nblocks((w + 15u) / 16u, (h + 15u) / 16u);

        cudaEvent_t ev_start, ev_stop;
        HANDLE_ERROR(cudaEventCreate(&ev_start));
        HANDLE_ERROR(cudaEventCreate(&ev_stop));
        HANDLE_ERROR(cudaEventRecord(ev_start));

        kernel<<< nblocks, nthreads >>>(dev_spheres, nb_spheres, w, h, dev_Ls, nb_samples);

        HANDLE_ERROR(cudaEventRecord(ev_stop));
        HANDLE_ERROR(cudaEventSynchronize(ev_stop));
        float ms = 0.0f;
        HANDLE_ERROR(cudaEventElapsedTime(&ms, ev_start, ev_stop));

        // Primary rays: w*h pixels * 4 subpixels * nb_samples per subpixel
        const double primary_rays = (double)w * h * 4.0 * nb_samples;
        const double mrays_sec = primary_rays / (ms * 1e3); // ms->s, rays->Mrays

        const std::uint32_t effective_spp = nb_samples * 4u;
        printf("Resolution: %ux%u  SPP: %u  Time: %.2f ms  Mrays/s: %.2f\n",
               w, h, effective_spp, ms, mrays_sec);
        printf("CSV: cuda_baseline,%ux%u,%u,%.2f,%.2f,sm_86_fp32\n",
               w, h, effective_spp, ms, mrays_sec);

        Vector3* Ls = (Vector3*)malloc(nb_pixels * sizeof(Vector3));
        HANDLE_ERROR(cudaMemcpy(Ls, dev_Ls, nb_pixels * sizeof(Vector3), cudaMemcpyDeviceToHost));

        HANDLE_ERROR(cudaFree(dev_Ls));
        HANDLE_ERROR(cudaFree(dev_spheres));
        HANDLE_ERROR(cudaEventDestroy(ev_start));
        HANDLE_ERROR(cudaEventDestroy(ev_stop));

        WritePPM(w, h, Ls, output_path);
        printf("Wrote: %s\n", output_path);

        free(Ls);
    }

} // namespace smallpt

static void usage(const char* prog) {
    fprintf(stderr, "Usage: %s [--width W] [--height H] [--spp N] [--output FILE]\n", prog);
    fprintf(stderr, "  Defaults: 1024x768, 40 spp, cu-image.ppm\n");
    fprintf(stderr, "  Note: SPP is rounded up to nearest multiple of 4 (2x2 subpixel sampling)\n");
}

int main(int argc, char* argv[]) {
    std::uint32_t w      = 1024u;
    std::uint32_t h      = 768u;
    std::uint32_t spp    = 40u;
    const char*   output = "cu-image.ppm";

    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--width")  == 0 && i + 1 < argc) { w      = (std::uint32_t)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) { h      = (std::uint32_t)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--spp")    == 0 && i + 1 < argc) { spp    = (std::uint32_t)atoi(argv[++i]); }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { output = argv[++i]; }
        else if (strcmp(argv[i], "--help")   == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown argument: %s\n", argv[i]); usage(argv[0]); return 1; }
    }

    // nb_samples is per-subpixel; effective SPP = nb_samples * 4
    const std::uint32_t nb_samples = (spp + 3u) / 4u;
    smallpt::Render(w, h, nb_samples, output);
    return 0;
}

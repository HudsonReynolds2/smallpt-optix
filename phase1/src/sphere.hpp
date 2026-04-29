#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "geometry.cuh"

#pragma endregion

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

// Bumped from 1e-4f to 1e-2f (Fix A). Float quantization on `t` for the
// 1e5-radius wall spheres is ~1mm at the camera, so a 1e-4 offset is
// smaller than the noise floor and self-intersection happens regularly.
// 1e-2 (1cm in scene units) is well above the quantization step but still
// invisible at the camera distance.
#define EPSILON_SPHERE 1e-2f

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	//-------------------------------------------------------------------------
	// Declarations and Definitions: Reflection_t
	//-------------------------------------------------------------------------

	enum struct Reflection_t {
		Diffuse,
		Specular,
		Refractive
	};

	//-------------------------------------------------------------------------
	// Declarations and Definitions: Sphere
	//-------------------------------------------------------------------------

	struct Sphere {

		//---------------------------------------------------------------------
		// Constructors and Destructors
		//---------------------------------------------------------------------

		__host__ __device__ explicit Sphere(float r,
		                                    Vector3 p,
		                                    Vector3 e,
		                                    Vector3 f,
		                                    Reflection_t reflection_t) noexcept
			: m_r(r),
			m_p(std::move(p)),
			m_e(std::move(e)),
			m_f(std::move(f)),
			m_reflection_t(reflection_t) {}
		Sphere(const Sphere& sphere) noexcept = default;
		Sphere(Sphere&& sphere) noexcept = default;
		~Sphere() = default;

		//---------------------------------------------------------------------
		// Assignment Operators
		//---------------------------------------------------------------------

		Sphere& operator=(const Sphere& sphere) = default;
		Sphere& operator=(Sphere&& sphere) = default;

		//---------------------------------------------------------------------
		// Member Methods
		//---------------------------------------------------------------------

		// Numerically stable ray-sphere intersection (Fix B).
		//
		// Original smallpt uses the algebraic form:
		//     D = (d.op)^2 - (op.op - r*r)
		// In fp32 with op ~ 1e5 and r ~ 1e5, both `(d.op)^2` and `(op.op - r*r)`
		// are ~1e10 and their difference is the actually-meaningful quantity
		// (~r*r * sin(theta)^2). Catastrophic cancellation eats most of the
		// mantissa, so `t` ends up quantized to ~1mm steps at the wall
		// distance.
		//
		// Geometric form: decompose `op = m_p - ray.m_o` into components
		// parallel and perpendicular to the ray direction:
		//     op_par  = (d . op) * d
		//     op_perp = op - op_par
		//     D       = r*r - |op_perp|^2
		// Because |op_perp| <= r whenever the ray hits, all squared
		// quantities stay well below 1e10 and we keep all 24 mantissa bits.
		//
		// `n_out` receives the *unnormalized* outward surface normal at the
		// hit point. Derivation: at hit parameter t, the hit point is
		// `p = ray.m_o + t*d`, and the vector from sphere center to hit is
		//     p - m_p = (ray.m_o + t*d) - m_p
		//             = -op + t*d
		//             = (t - dop)*d - perp
		// where t-dop = -sqrtD for the near hit and +sqrtD for the far hit.
		// All terms here are bounded by r, so this avoids the ~1e5 - ~1e5
		// subtraction that `Normalize(p - m_p)` would do in the kernel
		// (Fix C). Caller normalizes.
		__device__ bool Intersect(const Ray& ray, Vector3& n_out) const {
			const Vector3 op   = m_p - ray.m_o;
			const float dop    = ray.m_d.Dot(op);
			const Vector3 perp = op - ray.m_d * dop;
			const float perp2  = perp.Dot(perp);
			const float r2     = m_r * m_r;
			const float D      = r2 - perp2;

			if (D < 0.0f) {
				return false;
			}

			const float sqrtD = sqrtf(D);

			const float tmin = dop - sqrtD;
			if (ray.m_tmin < tmin && tmin < ray.m_tmax) {
				ray.m_tmax = tmin;
				// (t - dop)*d - perp with t - dop = -sqrtD
				n_out = ray.m_d * (-sqrtD) - perp;
				return true;
			}

			const float tmax = dop + sqrtD;
			if (ray.m_tmin < tmax && tmax < ray.m_tmax) {
				ray.m_tmax = tmax;
				// (t - dop)*d - perp with t - dop = +sqrtD
				n_out = ray.m_d * sqrtD - perp;
				return true;
			}

			return false;
		}

		// Convenience overload for callers that don't need the normal.
		__device__ bool Intersect(const Ray& ray) const {
			Vector3 unused;
			return Intersect(ray, unused);
		}

		//---------------------------------------------------------------------
		// Member Variables
		//---------------------------------------------------------------------

		float m_r;
		Vector3 m_p; // position
		Vector3 m_e; // emission
		Vector3 m_f; // reflection
		Reflection_t m_reflection_t;
	};
}

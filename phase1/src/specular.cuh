#pragma once

//-----------------------------------------------------------------------------
// CUDA Includes
//-----------------------------------------------------------------------------
#pragma region

#include "curand_kernel.h"

#pragma endregion

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "vector.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	__device__ inline float Reflectance0(float n1, float n2) {
		const float sqrt_R0 = (n1 - n2) / (n1 + n2);
		return sqrt_R0 * sqrt_R0;
	}

	__device__ inline float SchlickReflectance(float n1,
	                                           float n2,
	                                           float c) {

		const float R0 = Reflectance0(n1, n2);
		return R0 + (1.0f - R0) * c * c * c * c * c;
	}

	__device__ inline const Vector3 IdealSpecularReflect(const Vector3& d,
	                                                     const Vector3& n) {
		return d - 2.0f * n.Dot(d) * n;
	}

	__device__ inline const Vector3 IdealSpecularTransmit(const Vector3& d,
	                                                      const Vector3& n,
	                                                      float n_out,
	                                                      float n_in,
	                                                      float& pr,
	                                                      curandState* state) {

		const Vector3 d_Re = IdealSpecularReflect(d, n);

		const bool out_to_in = (0.0f > n.Dot(d));
		const Vector3 nl = out_to_in ? n : -n;
		const float nn = out_to_in ? n_out / n_in : n_in / n_out;
		const float cos_theta = d.Dot(nl);
		const float cos2_phi = 1.0f - nn * nn * (1.0f - cos_theta * cos_theta);

		// Total Internal Reflection
		if (0.0f > cos2_phi) {
			pr = 1.0f;
			return d_Re;
		}

		const Vector3 d_Tr = Normalize(nn * d - nl * (nn * cos_theta + sqrtf(cos2_phi)));
		const float c = 1.0f - (out_to_in ? -cos_theta : d_Tr.Dot(n));

		const float Re = SchlickReflectance(n_out, n_in, c);
		const float p_Re = 0.25f + 0.5f * Re;
		if (curand_uniform(state) < p_Re) {
			pr = (Re / p_Re);
			return d_Re;
		}
		else {
			const float Tr = 1.0f - Re;
			const float p_Tr = 1.0f - p_Re;
			pr = (Tr / p_Tr);
			return d_Tr;
		}
	}
}

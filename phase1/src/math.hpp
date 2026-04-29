#pragma once

//-----------------------------------------------------------------------------
// CUDA Includes
//-----------------------------------------------------------------------------
#pragma region

#include "device_launch_parameters.h"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	//-------------------------------------------------------------------------
	// Constants
	//-------------------------------------------------------------------------

	constexpr float g_pi = 3.14159265358979323846f;

	//-------------------------------------------------------------------------
	// Utilities
	//-------------------------------------------------------------------------

	__host__ __device__ inline float Clamp(float v,
	                                       float low  = 0.0f,
	                                       float high = 1.0f) noexcept {

		return fminf(fmaxf(v, low), high);
	}

	inline std::uint8_t ToByte(float color, float gamma = 2.2f) noexcept {
		const float gcolor = std::pow(color, 1.0f / gamma);
		return static_cast< std::uint8_t >(Clamp(255.0f * gcolor, 0.0f, 255.0f));
	}
}

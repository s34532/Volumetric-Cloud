/*
 * VolumetricCloudShader - GPU-Accelerated Volumetric Cloud Raymarching
 *
 * Implements a weather-driven atmospheric cloudscape using raymarching with:
 * - Layered low-cloud volume plus projected high cirrus
 * - Weather coverage, height profiles, and two-level Perlin-Worley density
 * - Beer's Law for light absorption
 * - Dual Henyey-Greenstein scattering and cone-traced self-shadowing
 * - DirectCompute GPU acceleration
 */

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEGP_SuiteHandler.h"
#include <math.h>
#include <stdlib.h>

// DirectX 11 for GPU compute
#include <d3d11.h>
#include <d3dcompiler.h>
#include <algorithm>
#include <mutex>
#include <string>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

// Thread synchronization for GPU resources
static std::mutex g_gpuMutex;

// Global GPU resources
static ID3D11Device* g_device = nullptr;
static ID3D11DeviceContext* g_context = nullptr;
static ID3D11ComputeShader* g_computeShader = nullptr;
static ID3D11ComputeShader* g_noiseShader = nullptr;
static ID3D11ComputeShader* g_sharpenShader = nullptr;
static ID3D11Buffer* g_constantBuffer = nullptr;
static ID3D11Texture2D* g_inputTex = nullptr;
static ID3D11Texture2D* g_outputTex = nullptr;
static ID3D11Texture2D* g_stagingTex = nullptr;
static ID3D11UnorderedAccessView* g_inputUAV = nullptr;
static ID3D11UnorderedAccessView* g_outputUAV = nullptr;
static ID3D11Texture3D* g_noiseTex = nullptr;
static ID3D11ShaderResourceView* g_noiseSRV = nullptr;
static ID3D11UnorderedAccessView* g_noiseUAV = nullptr;
static ID3D11SamplerState* g_noiseSampler = nullptr;
static int g_cachedWidth = 0;
static int g_cachedHeight = 0;
static bool g_gpuInitialized = false;
static bool g_gpuFailed = false;

// HLSL atmospheric cloud renderer. Its shape/lighting split follows the
// production techniques published by Rockstar and Guerrilla at SIGGRAPH.
static const char* g_shaderCodeChunks[] = { R"(
cbuffer CloudParams : register(b0) {
    float cloud_radius;
    float cloud_density;
    float absorption;
    float scattering;
    int march_steps;
    int light_steps;
    float sun_x, sun_y, sun_z;
    float sun_r, sun_g, sun_b;
    float wind_speed;
    float noise_scale;
    int detail;
    float camera_dist;
    float time;
    int width;
    int height;
    float camera_pitch;
    float camera_yaw;
    float wind_direction_x;
    float wind_direction_z;
    float wind_shear;
    float wind_turbulence;
    float cloud_coverage;
    float cloud_type;
    float erosion_strength;
    float billow_size;
    float mid_level_amount;
    float high_level_amount;
    float storm_development;
    int cloud_regime;
    float low_level_amount;
    float mid_cellular_style;
    float low_coverage_offset;
    float mid_coverage_offset;
    float tower_development;
    float tower_scale;
    float cauliflower_detail;
    float tower_position_x;
    float tower_distance;
    float tower_isolation;
    float tower_padding_0;
    int rain_enabled;
    float rain_amount;
    float rain_prevalence;
    float rain_shaft_detail;
    float rain_mist;
    float rain_evaporation;
    float rain_fall_speed;
    float rain_regime_factor;
    float sky_vibrancy;
    float sky_exposure;
    float twilight_intensity;
    float twilight_range;
    float horizon_haze;
    float sun_disk_size;
    float sun_brightness;
    float night_brightness;
    int stars_enabled;
    float star_amount;
    float star_brightness;
    float star_twinkle;
    int moon_enabled;
    float moon_azimuth;
    float moon_elevation;
    float moon_size;
    float moon_phase;
    float moon_brightness;
    float moon_glow;
    float moon_drift_speed;
    float moon_surface_detail;
    float moon_earthshine;
    float sky_padding_0;
    float sky_padding_1;
};

RWTexture2D<float4> InputTexture : register(u0);
RWTexture2D<float4> OutputTexture : register(u1);
RWTexture3D<float4> NoiseVolume : register(u2);
Texture3D<float4> CloudNoise : register(t0);
SamplerState CloudNoiseSampler : register(s0);

#define PI 3.14159265359f
#define EPSILON 0.0001f
#define NOISE_VOLUME_SIZE 128

float HenyeyGreenstein(float g, float mu) {
    float gg = g * g;
    return (1.0f / (4.0f * PI)) * ((1.0f - gg) / pow(1.0f + gg - 2.0f * g * mu, 1.5f));
}

// Smooth hash function for noise - quintic interpolation for softer edges
float3 hash33(float3 p) {
    p = float3(dot(p, float3(127.1f, 311.7f, 74.7f)),
               dot(p, float3(269.5f, 183.3f, 246.1f)),
               dot(p, float3(113.5f, 271.9f, 124.6f)));
    return frac(sin(p) * 43758.5453f);
}

float noise3d(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);

    // Quintic interpolation for smoother results (less sharp edges)
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    // Gradient noise for smoother appearance
    float3 ga = hash33(i + float3(0.0f, 0.0f, 0.0f)) * 2.0f - 1.0f;
    float3 gb = hash33(i + float3(1.0f, 0.0f, 0.0f)) * 2.0f - 1.0f;
    float3 gc = hash33(i + float3(0.0f, 1.0f, 0.0f)) * 2.0f - 1.0f;
    float3 gd = hash33(i + float3(1.0f, 1.0f, 0.0f)) * 2.0f - 1.0f;
    float3 ge = hash33(i + float3(0.0f, 0.0f, 1.0f)) * 2.0f - 1.0f;
    float3 gf = hash33(i + float3(1.0f, 0.0f, 1.0f)) * 2.0f - 1.0f;
    float3 gg = hash33(i + float3(0.0f, 1.0f, 1.0f)) * 2.0f - 1.0f;
    float3 gh = hash33(i + float3(1.0f, 1.0f, 1.0f)) * 2.0f - 1.0f;

    float va = dot(ga, f - float3(0.0f, 0.0f, 0.0f));
    float vb = dot(gb, f - float3(1.0f, 0.0f, 0.0f));
    float vc = dot(gc, f - float3(0.0f, 1.0f, 0.0f));
    float vd = dot(gd, f - float3(1.0f, 1.0f, 0.0f));
    float ve = dot(ge, f - float3(0.0f, 0.0f, 1.0f));
    float vf = dot(gf, f - float3(1.0f, 0.0f, 1.0f));
    float vg = dot(gg, f - float3(0.0f, 1.0f, 1.0f));
    float vh = dot(gh, f - float3(1.0f, 1.0f, 1.0f));

    return lerp(lerp(lerp(va, vb, u.x), lerp(vc, vd, u.x), u.y),
                lerp(lerp(ve, vf, u.x), lerp(vg, vh, u.x), u.y), u.z);
}

// Normalized, softly rotating FBM. Keeping its output in 0..1 makes the
// density controls predictable and prevents negative optical depth.
float fbm01(float3 p, int octaveCount) {
    float sum = 0.0f;
    float amplitude = 0.5f;
    float normalization = 0.0f;
    float3 q = p;

    [loop]
    for (int i = 0; i < 8; ++i) {
        if (i >= octaveCount) break;
        sum += (noise3d(q) * 0.5f + 0.5f) * amplitude;
        normalization += amplitude;
        q = float3(
            q.x * 1.83f + q.z * 0.68f,
            q.y * 2.03f,
            q.z * 1.83f - q.x * 0.68f
        );
        amplitude *= 0.5f;
    }

    return sum / max(normalization, EPSILON);
}

// Cheap analytic 2D value noise for projected ice clouds. A fixed four-octave
// sum avoids the contour/ringing artifacts produced by stretching one slice of
// the baked 3D Perlin-Worley volume across the entire upper sky.
float hash12(float2 p) {
    float3 p3 = frac(float3(p.x, p.y, p.x) * 0.1031f);
    p3 += dot(p3, p3.yzx + 33.33f);
    return frac((p3.x + p3.y) * p3.z);
}

float valueNoise2d(float2 p) {
    float2 cell = floor(p);
    float2 f = frac(p);
    float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    float a = hash12(cell);
    float b = hash12(cell + float2(1.0f, 0.0f));
    float c = hash12(cell + float2(0.0f, 1.0f));
    float d = hash12(cell + float2(1.0f, 1.0f));
    return lerp(lerp(a, b, u.x), lerp(c, d, u.x), u.y);
}

float fbm2d4(float2 p) {
    float result = valueNoise2d(p) * 0.533333f;
    p = float2(p.x * 1.83f + p.y * 0.68f,
               p.y * 1.83f - p.x * 0.68f) + 11.7f;
    result += valueNoise2d(p) * 0.266667f;
    p = float2(p.x * 1.83f + p.y * 0.68f,
               p.y * 1.83f - p.x * 0.68f) + 7.3f;
    result += valueNoise2d(p) * 0.133333f;
    p = float2(p.x * 1.83f + p.y * 0.68f,
               p.y * 1.83f - p.x * 0.68f) + 19.1f;
    result += valueNoise2d(p) * 0.066667f;
    return saturate(result);
}

float remapClamped(float value, float oldMin, float oldMax) {
    return saturate((value - oldMin) / max(oldMax - oldMin, EPSILON));
}

// Arithmetic hash used by cellular noise. Unlike the gradient-noise hash this
// avoids trigonometry because Worley evaluates many neighboring feature cells.
float3 hash33Fast(float3 p) {
    p = frac(p * float3(0.1031f, 0.1030f, 0.0973f));
    p += dot(p, p.yxz + 33.33f);
    return frac((p.xxy + p.yxx) * p.zyx);
}

// Exact 3x3x3 Worley F1 search. Inverted/combined cellular fields create the
// tightly packed round billows that plain Perlin FBM cannot produce.
float worleyF1(float3 p) {
    float3 lattice = floor(p);
    float3 local = frac(p);
    float minimumDistance2 = 8.0f;

    [unroll]
    for (int z = -1; z <= 1; ++z) {
        [unroll]
        for (int y = -1; y <= 1; ++y) {
            [unroll]
            for (int x = -1; x <= 1; ++x) {
                float3 neighbor = float3((float)x, (float)y, (float)z);
                float3 feature = hash33Fast(lattice + neighbor);
                float3 delta = neighbor + feature - local;
                minimumDistance2 = min(minimumDistance2, dot(delta, delta));
            }
        }
    }

    return saturate(sqrt(minimumDistance2) * 0.86f);
}

// Rockstar's atmosphere is made from nested world-scale volumes.  This effect
// uses the same important structural idea: the camera looks into a cloud shell,
// while a slow 2D weather field controls coverage and cloud type.  cloud_radius
// is retained in the parameter ABI but now means low-cloud layer thickness.
float cloudLayerBase() {
    return 1.45f;
}

// Low / middle / high amount plus vertical storm development. The host resolves
// each preset once per frame so the density loops remain branch-free.
float4 cloudDeckProfile() {
    return saturate(float4(low_level_amount, mid_level_amount,
                           high_level_amount, storm_development));
}

float selectedLowCloudType() {
    return cloud_type;
}

float baseCloudLayerTop() {
    return cloudLayerBase() + max(cloud_radius * 1.15f, 0.55f);
}

float cloudLayerTop() {
    // Explosive towers extend the conservative ray bounds, but their actual
    // local top is still masked by the weather field inside sampleDensity.
    return baseCloudLayerTop() + cloudDeckProfile().w * 4.2f
         + max(tower_development, 0.0f) * 8.4f;
}

float cloudHeight01(float3 p) {
    return saturate((p.y - cloudLayerBase())
                  / max(cloudLayerTop() - cloudLayerBase(), EPSILON));
}

float2 windDirectionXZ() {
    return float2(wind_direction_x, wind_direction_z);
}

float smoothTriangle(float phase) {
    float waveValue = 1.0f - 4.0f * abs(frac(phase) - 0.5f);
    return waveValue * (1.5f - 0.5f * abs(waveValue));
}

float cloudAnimationTime() {
    // "time" is in seconds. Keeping speed out of the CPU-side clock makes the
    // control linear and lets every cloud scale advect from the same timeline.
    return time * max(wind_speed, 0.0f);
}

// World-space displacement is essential for correct range cues. The old code
// subtracted UV offsets, making the high projected layer travel tens of times
// faster in physical space. Altitude changes wind speed, while perspective now
// naturally makes distant/high clouds cross the view more slowly.
float3 animatedCloudOffset(float3 p, float height01, float rate) {
    float animationTime = cloudAnimationTime();
    float2 direction = windDirectionXZ();
    float2 crossWind = float2(-direction.y, direction.x);
    float altitudeSpeed = lerp(0.018f, 0.031f, height01);
    float shear = (height01 - 0.32f) * wind_shear;
    float gust = smoothTriangle(dot(p.xz, float2(0.0382f, 0.0271f))
                              + animationTime * 0.0493f + height01 * 0.8435f);
    float2 horizontal = direction * animationTime * altitudeSpeed
                      + crossWind * (animationTime * shear * 0.0032f
                      + gust * wind_turbulence * 0.045f);
    float vertical = animationTime * wind_turbulence * 0.0012f
                   + smoothTriangle(animationTime * 0.0366f
                                  + height01 * 0.9549f)
                   * wind_turbulence * 0.022f;
    return float3(horizontal.x, vertical, horizontal.y) * rate;
}

// Two widely separated texture slices act as a low-frequency weather map.
// R controls regional coverage, G precipitation/density, and B cloud type.
float3 sampleWeather(float2 worldXZ) {
    float weatherScale = 0.0125f / max(noise_scale * 0.35f + 0.65f, 0.25f);
    float3 weatherWind = animatedCloudOffset(
        float3(worldXZ.x, cloudLayerBase(), worldXZ.y), 0.25f, 0.72f);
    float2 uv = (worldXZ - weatherWind.xz) * weatherScale;
    float4 broad = CloudNoise.SampleLevel(
        CloudNoiseSampler, float3(uv.x, 0.173f, uv.y), 0.0f);
    float4 regional = CloudNoise.SampleLevel(
        CloudNoiseSampler,
        float3(uv.x * 2.07f + 0.31f, 0.683f, uv.y * 2.07f - 0.19f), 0.0f);

    float coverage = saturate(broad.a * 0.63f + broad.r * 0.24f
                            + regional.a * 0.13f);
    coverage = smoothstep(0.30f, 0.76f, coverage);
    float precipitation = smoothstep(0.57f, 0.88f,
        broad.r * 0.58f + regional.g * 0.42f);
    float cloudType = smoothstep(0.16f, 0.67f,
        broad.g * 0.45f + regional.r * 0.55f);
    return float3(coverage, precipitation, cloudType);
}

float densityHeightGradient(float height01, float cloudType) {
    float stratus = smoothstep(0.025f, 0.10f, height01)
                  * (1.0f - smoothstep(0.56f, 0.67f, height01));
    float stratocumulus = smoothstep(0.015f, 0.095f, height01)
                        * (1.0f - smoothstep(0.76f, 0.88f, height01));
    float cumulus = smoothstep(0.01f, 0.075f, height01)
                  * (1.0f - smoothstep(0.90f, 0.995f, height01));
    float lowType = lerp(stratus, stratocumulus, saturate(cloudType * 1.6f));
    return lerp(lowType, cumulus, smoothstep(0.48f, 0.92f, cloudType));
}

float sampleLightingDetail(float3 p) {
    float height01 = cloudHeight01(p);
    float frequency = max(noise_scale / max(billow_size, 0.25f), 0.05f);
    float3 advected = p - animatedCloudOffset(p, height01, 1.58f);
    float3 uv = advected * float3(0.18f, 0.30f, 0.18f) * frequency;
    float4 detailNoise = CloudNoise.SampleLevel(CloudNoiseSampler, uv, 0.0f);
    return saturate(detailNoise.g * 0.44f + detailNoise.b * 0.56f);
}

)",
R"(
// Production-style two-level Nubis density model: weather and height define a
// coherent low-frequency body, then a separate cellular field erodes only its
// boundary.  There is deliberately no sphere, SDF, or hand-shaped silhouette.
float sampleDensity(float3 p, bool coarse) {
    if (p.y <= cloudLayerBase() || p.y >= cloudLayerTop()) return 0.0f;
    float3 weather = sampleWeather(p.xz);
    float4 decks = cloudDeckProfile();
    float convectiveColumn = smoothstep(
        0.42f, 0.82f, weather.x * 0.48f + weather.y * 0.52f);
    float towerAmount = max(tower_development, 0.0f);
    float widthControl = saturate((tower_scale - 0.25f) / 2.75f);
    float towerMoisture = weather.y * 0.62f + weather.x * 0.38f;
    float towerThreshold = lerp(0.72f, 0.52f, widthControl);
    float weatherTower = smoothstep(towerThreshold, 0.92f, towerMoisture);
    float3 towerWind = animatedCloudOffset(
        float3(tower_position_x, cloudLayerBase(), tower_distance),
        0.28f, 0.82f);
    float2 towerCenter = float2(tower_position_x, tower_distance) + towerWind.xz;
    float towerRadius = lerp(4.0f, 22.0f, widthControl);
    float2 towerRelative = (p.xz - towerCenter)
        / float2(towerRadius, towerRadius * 0.78f);
    float formTime = cloudAnimationTime() * 0.012f;
    float4 towerForm = float4(0.5f, 0.5f, 0.5f, 0.5f);
    if (towerAmount > 0.001f) {
        towerForm = CloudNoise.SampleLevel(CloudNoiseSampler,
            float3(towerRelative.x * 0.31f + 0.23f,
                   p.y * 0.052f + formTime,
                   towerRelative.y * 0.31f + 0.67f), 0.0f);
    }
    float formWarp = (towerForm.a - 0.5f) * 0.19f
                   + (0.52f - towerMoisture) * 0.13f;

    // A mature cell is a family of overlapping buoyant thermals rather than a
    // single capsule. Their shared base stays connected while their unequal
    // tops create the expansive multi-turret silhouette of deep convection.
    float mainThermal = 1.0f - smoothstep(0.28f, 0.82f,
        length(towerRelative - float2(-0.04f, 0.05f)) + formWarp);
    float leftThermal = 1.0f - smoothstep(0.13f, 0.64f,
        length((towerRelative - float2(-0.34f, 0.02f)) * float2(1.0f, 1.12f))
        - formWarp * 0.55f);
    float rightThermal = 1.0f - smoothstep(0.14f, 0.66f,
        length((towerRelative - float2(0.34f, 0.09f)) * float2(1.0f, 1.09f))
        + formWarp * 0.42f);
    float forwardThermal = 1.0f - smoothstep(0.12f, 0.58f,
        length((towerRelative - float2(0.13f, -0.28f)) * float2(1.09f, 1.0f))
        - formWarp * 0.35f);
    float heroFootprint = max(max(mainThermal, leftThermal),
                              max(rightThermal, forwardThermal));
    float heightFootprint = max(mainThermal,
        max(leftThermal * 0.76f,
            max(rightThermal * 0.86f, forwardThermal * 0.68f)));
    float towerRadial = length(towerRelative - float2(-0.04f, 0.05f));
    float towerCore = 1.0f - smoothstep(0.16f, 0.55f,
        towerRadial + formWarp * 0.32f);
    float towerColumn = max(weatherTower,
        heroFootprint * lerp(0.84f, 1.0f, towerMoisture))
        * saturate(towerAmount);
    float towerHeightField = saturate(max(weatherTower,
        heightFootprint * saturate(towerAmount))
        * lerp(0.82f, 1.18f, towerForm.r));
    float localTop = baseCloudLayerTop()
                   + decks.w * 4.2f * convectiveColumn
                   + towerAmount * 8.4f * towerHeightField;
    float height01 = saturate((p.y - cloudLayerBase())
        / max(localTop - cloudLayerBase(), EPSILON));
    if (p.y >= localTop || decks.x <= 0.001f) return 0.0f;

    float coverageBias = saturate(cloud_coverage + low_coverage_offset
        + (cloud_density - 1.3f) * 0.025f + (decks.x - 1.0f) * 0.58f);
    float isolationMask = lerp(1.0f,
        saturate(heroFootprint * 1.18f + 0.055f),
        saturate(tower_isolation) * saturate(towerAmount));
    float localCoverage = saturate(
        (coverageBias + (weather.x - 0.5f) * 0.44f) * isolationMask
        + towerColumn * towerAmount * 0.64f);
    if (localCoverage <= 0.08f) return 0.0f;

    float lowType = selectedLowCloudType();
    float frequency = max(noise_scale / max(billow_size, 0.25f), 0.05f)
                    * lerp(0.74f, 1.06f, lowType);
    frequency *= lerp(1.0f, lerp(1.18f, 0.62f, widthControl), towerColumn);
    float3 advectedBase = p - animatedCloudOffset(p, height01, 1.0f);
    float3 noiseUv = advectedBase * float3(0.095f, 0.22f, 0.095f) * frequency
                   + float3(0.18f, 0.11f, 0.29f);
    float4 warpNoise = CloudNoise.SampleLevel(
        CloudNoiseSampler, noiseUv * 0.47f + float3(0.265f, 0.41f, 0.265f), 0.0f);
    noiseUv += float3(
        warpNoise.a - 0.5f,
        (warpNoise.b - 0.5f) * 0.32f,
        warpNoise.a - warpNoise.b) * (0.14f + wind_turbulence * 0.075f);
    float4 noises = CloudNoise.SampleLevel(CloudNoiseSampler, noiseUv, 0.0f);

    float localCloudType = saturate(lowType * 0.76f + weather.z * 0.29f - 0.03f);
    float heightGradient = densityHeightGradient(height01, localCloudType);
    float perlinWorley = saturate((noises.r - 0.40f) * 1.38f + 0.50f);
    float baseShape = remapClamped(perlinWorley, 1.0f - localCoverage, 1.0f)
                    * heightGradient;
    // Stacked low-frequency pulses create connected cauliflower turrets. They
    // are a modulation of the weather-driven body, not a sphere/SDF silhouette.
    if (towerColumn > 0.001f) {
        float verticalLobes = smoothTriangle(
            height01 * lerp(3.2f, 7.4f, saturate(cauliflower_detail * 0.5f))
            + warpNoise.a * 0.72f) * 0.5f + 0.5f;
        float turretNoise = saturate(noises.r * 0.58f + warpNoise.r * 0.27f
                                   + verticalLobes * 0.15f);
        float turretBody = remapClamped(
            turretNoise, lerp(0.54f, 0.39f, towerColumn), 0.92f)
                         * heightGradient * towerColumn;
        baseShape = max(baseShape, turretBody * (0.78f + towerAmount * 0.34f));

        // A buoyant cumulonimbus has a continuous optically thick updraft
        // surrounded by noisy entrainment.  Keep that core low-frequency and
        // slightly heterogeneous; the later cellular erosion still sculpts
        // its surface, but can no longer punch sky-sized holes through it.
        float coreVariation = saturate(0.76f
            + (noises.r - 0.5f) * 0.28f
            + (warpNoise.r - 0.5f) * 0.16f);
        float connectedCore = towerCore * towerColumn * heightGradient
                            * coreVariation * saturate(towerAmount);
        baseShape = max(baseShape,
            connectedCore * lerp(0.86f, 1.12f, saturate(towerAmount - 0.25f)));
    }
    baseShape *= lerp(0.90f, 1.16f, weather.y);
    if (baseShape <= 0.002f) return 0.0f;

    if (coarse) {
        return saturate(baseShape) * lerp(0.72f, 1.0f, weather.y) * decks.x;
    }

    float detailFrequency = max(noise_scale
        * lerp(1.18f, 0.88f, saturate((billow_size - 0.25f) / 2.75f)), 0.05f);
    float3 detailWind = animatedCloudOffset(p, height01, 1.68f);
    float animationTime = cloudAnimationTime();
    float2 crossWind = float2(-windDirectionXZ().y, windDirectionXZ().x);
    float3 evolvingErosion = float3(crossWind.x, 0.82f, crossWind.y)
                          * animationTime * wind_turbulence * 0.0032f;
    float3 detailPosition = p - detailWind + evolvingErosion;
    float3 detailUv = detailPosition * float3(0.48f, 0.70f, 0.48f)
                    * detailFrequency + 0.37f;
    float4 details = CloudNoise.SampleLevel(CloudNoiseSampler, detailUv, 0.0f);
    float4 microDetails = CloudNoise.SampleLevel(
        CloudNoiseSampler,
        detailUv * 1.91f + float3(0.17f, 0.43f, 0.71f), 0.0f);
    float erosion = details.g * 0.42f + details.b * 0.23f
                  + microDetails.g * 0.20f + microDetails.b * 0.10f
                  + noises.b * 0.05f;
    float lowerWisps = 1.0f - smoothstep(0.10f, 0.28f, height01);
    erosion = lerp(erosion, 1.0f - erosion, lowerWisps * 0.74f);
    float edgeWeight = 1.0f - smoothstep(0.42f, 0.96f, baseShape);
    float detailStrength = lerp(0.48f, 0.82f, saturate((float)detail / 8.0f))
                         * max(erosion_strength, 0.0f);
    float protectedCore = towerCore * towerColumn * saturate(towerAmount)
        * lerp(1.0f, 0.42f, smoothstep(0.68f, 0.96f, height01));
    detailStrength *= lerp(1.0f, 0.42f, protectedCore);
    float density = remapClamped(baseShape, erosion * edgeWeight * detailStrength, 1.0f);
    float cauliflowerStrength = towerColumn * saturate(towerAmount)
                              * saturate(cauliflower_detail * 0.5f);
    if (cauliflowerStrength > 0.001f) {
        float cauliflowerCells = smoothstep(0.34f, 0.70f,
            details.r * 0.46f + microDetails.r * 0.24f
          + noises.r * 0.30f);
        float sculptedTower = remapClamped(
            baseShape + (cauliflowerCells - 0.48f) * 0.34f,
            0.10f, 0.88f);
        density = lerp(density, max(density, sculptedTower),
                       cauliflowerStrength * 0.78f);
    }
    density = max(density, protectedCore * heightGradient * 0.46f);
    // Water clouds have comparatively crisp optical boundaries even though
    // the interior integrates softly. This remap prevents translucent noise
    // from reading as a Gaussian-blurred smoke sprite.
    density = smoothstep(0.085f, 0.56f, density);
    density *= lerp(0.72f, 1.22f, weather.y);
    return density * max(cloud_density * 1.28f, 0.0f) * decks.x;
}

)",
R"(
// Temperate middle-level deck (roughly the WMO 2-7 km band in normalized
// scene units). Altocumulus uses cellular streets; altostratus/nimbostratus
// use a broad, optically continuous sheet. This is a distinct volume, not a
// rescaled copy of the low cloud layer.
float middleLayerBase() { return 5.15f; }
float middleLayerTop() {
    float nimbusExtension = step(0.24f, mid_coverage_offset) * 1.15f;
    return 6.75f + nimbusExtension;
}

float middleCellularStyle() {
    return mid_cellular_style;
}

float sampleMiddleDensity(float3 p, bool coarse) {
    float amount = cloudDeckProfile().y;
    if (amount <= 0.001f || p.y <= middleLayerBase()
        || p.y >= middleLayerTop()) return 0.0f;

    float height01 = saturate((p.y - middleLayerBase())
        / max(middleLayerTop() - middleLayerBase(), EPSILON));
    float cellular = middleCellularStyle();
    float3 advected = p - animatedCloudOffset(p, 0.58f, 1.24f);
    float frequency = max(noise_scale, 0.05f) * lerp(0.42f, 0.96f, cellular);
    float3 uv = advected * float3(0.096f, 0.31f, 0.096f) * frequency
              + float3(0.63f, 0.27f, 0.11f);
    float4 broad = CloudNoise.SampleLevel(CloudNoiseSampler, uv, 0.0f);
    float4 cells = CloudNoise.SampleLevel(
        CloudNoiseSampler, uv * 3.05f + float3(0.21f, 0.57f, 0.39f), 0.0f);

    // Reuse the middle deck's own broad sample as its weather field. Calling
    // sampleWeather here added two volume fetches at every middle ray step.
    float coverage = saturate(cloud_coverage * 0.72f + amount * 0.38f
                   + (broad.a - 0.5f) * 0.34f + mid_coverage_offset);
    if (coverage <= 0.10f) return 0.0f;

    float2 crossWind = float2(-windDirectionXZ().y, windDirectionXZ().x);
    float streetWave = smoothTriangle(dot(advected.xz, crossWind) * 0.048f
                                   + broad.a * 0.72f);
    float cloudStreet = saturate(streetWave * 0.46f + 0.72f);
    float cellCore = saturate(broad.r * 0.58f + cells.r * 0.42f);
    float cellularBody = remapClamped(
        cellCore, 1.03f - coverage, 0.94f)
                       * lerp(1.0f, cloudStreet, 0.64f);
    float sheetBody = remapClamped(
        broad.a * 0.48f + broad.r * 0.34f + cells.r * 0.18f,
        0.58f - coverage * 0.38f, 1.0f);
    float body = lerp(sheetBody, cellularBody, cellular);

    float cellularHeight = smoothstep(0.025f, 0.16f, height01)
                         * (1.0f - smoothstep(0.72f, 0.97f, height01));
    float sheetHeight = smoothstep(0.015f, 0.08f, height01)
                      * (1.0f - smoothstep(0.88f, 0.985f, height01));
    body *= lerp(sheetHeight, cellularHeight, cellular);
    if (body <= 0.002f) return 0.0f;
    if (coarse) return body * amount;

    float erosion = cells.g * 0.58f + cells.b * 0.42f;
    float edge = 1.0f - smoothstep(0.46f, 0.94f, body);
    body = remapClamped(body,
        erosion * edge * lerp(0.22f, 0.62f, cellular) * erosion_strength, 1.0f);
    body = smoothstep(0.07f, lerp(0.58f, 0.46f, cellular), body);
    float nimbusDensity = lerp(1.0f, 1.42f, step(0.24f, mid_coverage_offset));
    return body * amount * max(cloud_density * 0.72f, 0.0f) * nimbusDensity;
}

bool rayMiddleLayerBounds(float3 ro, float3 rd, out float nearT, out float farT) {
    if (abs(rd.y) < EPSILON || cloudDeckProfile().y <= 0.001f) {
        nearT = 0.0f;
        farT = 0.0f;
        return false;
    }
    float baseT = (middleLayerBase() - ro.y) / rd.y;
    float topT = (middleLayerTop() - ro.y) / rd.y;
    nearT = max(0.0f, min(baseT, topT));
    farT = min(max(0.0f, max(baseT, topT)), 58.0f);
    return farT > nearT;
}

bool rayCloudLayerBounds(float3 ro, float3 rd, out float nearT, out float farT) {
    if (abs(rd.y) < EPSILON) {
        nearT = 0.0f;
        farT = 0.0f;
        return false;
    }

    float baseT = (cloudLayerBase() - ro.y) / rd.y;
    float topT = (cloudLayerTop() - ro.y) / rd.y;
    nearT = max(0.0f, min(baseT, topT));
    farT = max(0.0f, max(baseT, topT));
    float horizonLimit = 38.0f + max(cloud_radius, 0.1f) * 8.0f;
    farT = min(farT, horizonLimit);
    return farT > nearT;
}

float lightmarch(float3 position, float3 sunDirection) {
    float lightNear, lightFar;
    if (!rayCloudLayerBounds(position, sunDirection, lightNear, lightFar)) {
        return 1.0f;
    }
    lightFar = min(lightFar, 22.0f);
    float opticalDepth = 0.0f;
    float3 helperAxis = abs(sunDirection.y) < 0.95f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = normalize(cross(sunDirection, helperAxis));
    float3 bitangent = cross(sunDirection, tangent);

    // Cone sampling gathers light from a small neighborhood instead of making
    // every shadow boundary follow one perfectly straight, overly soft ray.
    float previousTravel = 0.0f;
    [loop]
    for (int step = 0; step < light_steps; ++step) {
        float u = ((float)step + 1.0f) / max((float)light_steps, 1.0f);
        float travel = lightFar * u * u;
        float segmentLength = travel - previousTravel;
        float sampleTravel = previousTravel + segmentLength * 0.5f;
        float angle = (float)step * 2.399963f;
        float coneRadius = sampleTravel * 0.055f;
        float3 coneOffset = (tangent * cos(angle) + bitangent * sin(angle)) * coneRadius;
        float3 samplePosition = position + sunDirection * sampleTravel + coneOffset;
        bool cheapSample = step >= 3;
        opticalDepth += sampleDensity(samplePosition, cheapSample) * segmentLength;
        previousTravel = travel;
    }

    float extinction = max(absorption, 0.02f);
    float direct = exp(-opticalDepth * extinction * 1.18f);
    float scatter2 = exp(-opticalDepth * extinction * 0.28f);
    float scatter3 = exp(-opticalDepth * extinction * 0.065f);
    return direct * 0.61f + scatter2 * 0.29f + scatter3 * 0.10f;
}

)",
R"(
float4 raymarch(float3 rayOrigin, float3 rayDirection,
                float3 sunDirection, float jitter) {
    float nearT, farT;
    if (!rayCloudLayerBounds(rayOrigin, rayDirection, nearT, farT)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    // The conservative vertical ceiling of a deep tower can project far past
    // its localized horizontal footprint at low camera pitch. Distributing a
    // fixed sample count over that empty tail creates visible depth terraces.
    // Intersect the view ray with the hero updraft footprint and retain the
    // ordinary deck bound outside it, concentrating work where density exists.
    float heroRayHit = 0.0f;
    if (tower_development > 0.001f && tower_isolation > 0.25f) {
        float widthControl = saturate((tower_scale - 0.25f) / 2.75f);
        float towerRadius = lerp(4.0f, 22.0f, widthControl);
        float3 towerWind = animatedCloudOffset(
            float3(tower_position_x, cloudLayerBase(), tower_distance),
            0.28f, 0.82f);
        float2 towerCenter = float2(tower_position_x, tower_distance)
                           + towerWind.xz;
        float2 footprintRadii = float2(towerRadius, towerRadius * 0.78f) * 1.18f;
        float2 scaledOrigin = (rayOrigin.xz - towerCenter) / footprintRadii;
        float2 scaledDirection = rayDirection.xz / footprintRadii;
        float quadraticA = dot(scaledDirection, scaledDirection);
        float quadraticB = dot(scaledOrigin, scaledDirection);
        float quadraticC = dot(scaledOrigin, scaledOrigin) - 1.0f;
        float discriminant = quadraticB * quadraticB - quadraticA * quadraticC;

        float ordinaryTop = baseCloudLayerTop() + cloudDeckProfile().w * 4.2f;
        float ordinaryTopT = (ordinaryTop - rayOrigin.y) / rayDirection.y;
        float focusedFar = min(farT, max(nearT, ordinaryTopT));
        if (quadraticA > EPSILON && discriminant >= 0.0f) {
            float towerExit = (-quadraticB + sqrt(discriminant)) / quadraticA;
            if (towerExit > nearT) {
                focusedFar = max(focusedFar, min(farT, towerExit + 1.25f));
                heroRayHit = 1.0f;
            }
        } else if (quadraticC <= 0.0f) {
            // Near-vertical ray already inside the tower footprint.
            focusedFar = farT;
            heroRayHit = 1.0f;
        }
        float focusStrength = smoothstep(0.25f, 0.65f, tower_isolation)
                            * saturate(tower_development);
        farT = lerp(farT, min(farT, focusedFar), focusStrength);
    }

    float pathLength = farT - nearT;
    float towerStepTarget = lerp(
        0.46f, 0.30f, saturate(tower_development * 0.75f));
    int spatialSteps = (int)ceil(pathLength / max(towerStepTarget, 0.20f));
    int effectiveSteps = min(120, max(march_steps,
        (int)lerp((float)march_steps, (float)spatialSteps, heroRayHit)));
    float stepLength = pathLength / max((float)effectiveSteps, 1.0f);
    float depth = nearT + stepLength * jitter;
    float transmittance = 1.0f;
    float3 radiance = 0.0f;

    // Incoming light versus direction toward the viewer.
    // sunDirection points from the sample toward the sun. Looking in that
    // direction is the strong forward-scattering configuration.
    float mu = dot(rayDirection, sunDirection);
    float g = clamp(scattering, -0.85f, 0.85f);
    float phaseForward = HenyeyGreenstein(g, mu) * (4.0f * PI);
    float phaseBack = HenyeyGreenstein(-0.24f, mu) * (4.0f * PI);
    float phase = phaseForward * 0.78f + phaseBack * 0.22f;
    float daylight = smoothstep(-0.12f, 0.045f, sunDirection.y);
    float moonFill = moon_enabled != 0
        ? min(max(moon_brightness, 0.0f), 2.0f) * 0.22f : 0.0f;
    float3 nightAmbient = float3(0.035f, 0.052f, 0.11f)
        * (0.30f + max(night_brightness, 0.0f) * 1.8f + moonFill);
    float3 ambientTint = lerp(
        nightAmbient, float3(0.47f, 0.58f, 0.73f), daylight);
    float3 aerialTint = lerp(
        float3(0.018f, 0.032f, 0.075f)
            * (0.45f + max(night_brightness, 0.0f) + moonFill),
        float3(0.63f, 0.72f, 0.82f), daylight);

    [loop]
    for (int i = 0; i < effectiveSteps; ++i) {
        if (depth > farT || transmittance < 0.008f) break;
        float3 p = rayOrigin + rayDirection * depth;
        // The detailed sampler already exits after its weather/base checks in
        // empty space. Calling the coarse sampler first duplicated all of that
        // work for occupied samples without changing the step length.
        float density = sampleDensity(p, false);

        if (density > 0.001f) {
            float opticalStep = density * max(absorption, 0.02f) * stepLength * 1.12f;
            float sampleAlpha = 1.0f - exp(-opticalStep);
            float lightTransmittance = lightmarch(p, sunDirection);
            float height01 = cloudHeight01(p);
            float ambient = lerp(0.20f, 0.52f, height01);
            // Beer-powder response approximates the extra gathered light in
            // cloud interiors while retaining darker outward-facing edges.
            float powder = 1.0f - exp(-density * stepLength * 4.4f);
            float powderView = lerp(0.70f, 1.05f, saturate(mu * 0.5f + 0.5f));
            float localAhead = sampleDensity(
                p + sunDirection * 0.075f, false);
            float localVisibility = exp(
                -localAhead * max(absorption, 0.02f) * 0.68f);
            float localBillowLight = lerp(0.48f, 1.28f, localVisibility);
            float detailLight = lerp(0.58f, 1.42f, sampleLightingDetail(p));
            float3 sunColor = float3(sun_r, sun_g, sun_b);
            float3 lighting = sunColor * max(sun_brightness, 0.0f)
                            * lightTransmittance * phase
                            * localBillowLight * detailLight
                            * lerp(0.58f, 1.14f, powder) * powderView;
            lighting += ambientTint * ambient;
            lighting *= lerp(1.0f, 0.64f, step(0.18f, low_coverage_offset));

            // Distance aerial perspective makes the volume belong to the sky
            // instead of reading as a cut-out object against it.
            float aerial = exp(-depth * 0.018f);
            lighting = lerp(aerialTint, lighting, aerial);

            radiance += transmittance * sampleAlpha * lighting;
            transmittance *= exp(-opticalStep);
        }

        depth += stepLength;
    }

    return float4(radiance, transmittance);
}

float4 raymarchMiddle(float3 rayOrigin, float3 rayDirection,
                      float3 sunDirection, float jitter) {
    float nearT, farT;
    if (!rayMiddleLayerBounds(rayOrigin, rayDirection, nearT, farT)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    int middleSteps = max(5, min(10, march_steps / 12));
    float stepLength = (farT - nearT) / (float)middleSteps;
    float depth = nearT + stepLength * jitter;
    float transmittance = 1.0f;
    float3 radiance = 0.0f;
    float mu = dot(rayDirection, sunDirection);
    float phase = HenyeyGreenstein(clamp(scattering * 0.72f, -0.7f, 0.7f), mu)
                * (4.0f * PI);
    float daylight = smoothstep(-0.12f, 0.045f, sunDirection.y);
    float moonFill = moon_enabled != 0
        ? min(max(moon_brightness, 0.0f), 2.0f) * 0.18f : 0.0f;
    float3 middleAmbient = lerp(
        float3(0.030f, 0.044f, 0.095f)
            * (0.32f + max(night_brightness, 0.0f) * 1.7f + moonFill),
        float3(0.48f, 0.57f, 0.69f), daylight);
    float3 middleAerial = lerp(
        float3(0.020f, 0.033f, 0.070f)
            * (0.42f + max(night_brightness, 0.0f) + moonFill),
        float3(0.66f, 0.73f, 0.81f), daylight);

    [loop]
    for (int i = 0; i < middleSteps; ++i) {
        if (depth > farT || transmittance < 0.01f) break;
        float3 p = rayOrigin + rayDirection * depth;
        float density = sampleMiddleDensity(p, false);
        if (density > 0.001f) {
            float opticalStep = density * max(absorption, 0.02f)
                              * stepLength * 0.92f;
            float sampleAlpha = 1.0f - exp(-opticalStep);
            // Thin middle decks need depth integration, but a second cone
            // march is disproportionate. Local optical thickness supplies the
            // soft self-shadow expected from altostratus/altocumulus.
            float lightVisibility = exp(-density * max(absorption, 0.02f) * 0.34f);
            float cellular = middleCellularStyle();
            float3 sunColor = float3(sun_r, sun_g, sun_b);
            float3 lighting = sunColor * max(sun_brightness, 0.0f)
                            * lightVisibility * phase
                            * lerp(0.88f, 1.12f, cellular);
            lighting += middleAmbient * lerp(0.42f, 0.30f, cellular);
            float aerial = exp(-depth * 0.028f);
            lighting = lerp(middleAerial, lighting, aerial);
            radiance += transmittance * sampleAlpha * lighting;
            transmittance *= exp(-opticalStep);
        }
        depth += stepLength;
    }
    return float4(radiance, transmittance);
}

// One-time procedural volume bake. R stores connected Perlin-Worley macro
// shape; G/B store independent cellular/ridged erosion scales; A stores a
// low-frequency warp field. The cloud renderer samples this volume linearly.
[numthreads(4, 4, 4)]
void CSNoise(uint3 DTid : SV_DispatchThreadID) {
    if (any(DTid >= (uint3)NOISE_VOLUME_SIZE)) return;

    float3 uvw = (float3(DTid) + 0.5f) / (float)NOISE_VOLUME_SIZE;
    float3 p = uvw * 8.0f;

    float perlinRaw = fbm01(p * 0.46f + float3(1.7f, -3.1f, 4.6f), 5);
    float perlin = saturate((perlinRaw - 0.5f) * 2.05f + 0.5f);
    // Perlin-Worley macro structure: connected Perlin masses are carved by a
    // three-octave cellular FBM. A single Worley octave produced smooth blobs;
    // nested cells produce the cauliflower hierarchy visible in real cumulus.
    float baseCell0 = 1.0f - worleyF1(p * 0.58f + float3(5.3f, -2.4f, 1.1f));
    float baseCell1 = 1.0f - worleyF1(p * 1.16f + float3(-1.7f, 6.2f, 3.9f));
    float baseCell2 = 1.0f - worleyF1(p * 2.32f + float3(8.1f, 0.6f, -4.3f));
    float baseWorleyFbm = baseCell0 * 0.625f
                        + baseCell1 * 0.250f
                        + baseCell2 * 0.125f;
    float perlinWorley = remapClamped(perlin,
        0.62f - baseWorleyFbm * 0.65f, 1.0f);

    // Independent high-frequency Worley FBM is reserved for boundary erosion.
    // It remains in the baked texture, so the richer shape costs no additional
    // procedural-noise ALU during the per-frame ray march.
    float detailCell0 = 1.0f - worleyF1(p * 1.72f + float3(-7.1f, 2.8f, 9.4f));
    float detailCell1 = 1.0f - worleyF1(p * 3.44f + float3(2.3f, -8.7f, 5.1f));
    float detailCell2 = 1.0f - worleyF1(p * 6.88f + float3(11.2f, 4.6f, -3.8f));
    float cellularErosion = detailCell0 * 0.625f
                          + detailCell1 * 0.250f
                          + detailCell2 * 0.125f;
    float microCell = 1.0f - worleyF1(
        p * 5.15f + float3(-5.7f, 13.1f, 7.4f));
    float ridgedDetail = 1.0f - abs(noise3d(
        p * 7.4f + float3(-3.0f, 15.0f, 21.0f)));
    float warp = fbm01(p * 0.31f + float3(8.0f, 1.0f, -6.0f), 3);
    NoiseVolume[DTid] = float4(
        perlinWorley,
        saturate(cellularErosion),
        saturate(microCell * 0.68f + ridgedDetail * 0.32f),
        saturate(warp));
}

)",
R"(
// Thin upper-atmosphere clouds do not need a full volume march.  Projecting a
// stretched, ridged noise field onto a high plane follows the same low/high
// layer split used by production real-time sky renderers.
float sampleCirrus(float3 rayOrigin, float3 rayDirection) {
    float amount = cloudDeckProfile().z;
    if (rayDirection.y <= 0.035f || amount <= 0.001f) return 0.0f;
    const float highAltitude = 10.5f;
    float travel = highAltitude / rayDirection.y;
    float2 worldXZ = rayOrigin.xz + rayDirection.xz * travel;
    float animationTime = cloudAnimationTime();
    float2 direction = windDirectionXZ();
    float2 crossWind = float2(-direction.y, direction.x);
    float3 highWind = animatedCloudOffset(
        float3(worldXZ.x, highAltitude, worldXZ.y), 1.0f, 1.52f);
    float2 advectedWorld = worldXZ - highWind.xz;
    // Wind-aligned coordinates produce long fibratus trails. Because motion is
    // applied to worldXZ before projection, the 10.5-unit layer moves more
    // slowly on screen than nearby cumulus despite a stronger upper wind.
    float alongWind = dot(advectedWorld, direction);
    float acrossWind = dot(advectedWorld, crossWind);
    float flow = fbm2d4(float2(alongWind, acrossWind) * 0.052f + 4.7f);
    // Slowly varying domain curl breaks the old ruler-straight parallel bands.
    // The displacement is in world units, so the fibers bend without sliding
    // or changing speed as camera pitch changes.
    float meander = sin(alongWind * 0.021f + flow * (2.0f * PI)) * 0.58f
                   + (flow - 0.5f) * 1.25f;
    alongWind += (flow - 0.5f) * 7.0f;
    acrossWind += meander * 10.5f * (0.72f + wind_turbulence * 0.28f);
    float broad = fbm2d4(float2(alongWind * 0.12f, acrossWind * 0.35f));
    float strands = fbm2d4(float2(alongWind * 0.25f,
                                  acrossWind * 1.10f) + 23.4f);
    float micro = fbm2d4(float2(alongWind * 0.58f,
                                acrossWind * 2.35f) + 47.9f);
    float cirrusBody = broad * 0.58f + strands * 0.42f;
    float erodedCirrus = cirrusBody - (1.0f - micro) * 0.18f;
    float cirrusCore = smoothstep(0.40f, 0.60f, erodedCirrus);
    float cirrusFeather = smoothstep(0.38f, 0.64f, cirrusBody) * 0.28f;
    float streamMask = smoothstep(0.31f, 0.61f, broad);
    float cirrus = streamMask * (cirrusCore * 0.76f + cirrusFeather);
    float cirrostratus = smoothstep(0.22f, 0.66f, broad) * 0.88f;
    float2 mackerelUv = float2(alongWind, acrossWind) * 0.019f;
    float4 smallCells = CloudNoise.SampleLevel(
        CloudNoiseSampler, float3(mackerelUv.x, 0.448f, mackerelUv.y), 0.0f);
    float rowWave = smoothTriangle(acrossWind * 0.115f + smallCells.a * 0.48f);
    float cirrocumulus = smoothstep(0.42f, 0.64f,
                           smallCells.r * 0.58f + smallCells.g * 0.42f)
                       * smoothstep(-0.14f, 0.68f, rowWave);

    float cirrusWeight = 0.10f;
    float veilWeight = 0.22f;
    float rippleWeight = 0.16f;
    if (cloud_regime == 2) {
        cirrusWeight = 1.0f; veilWeight = 0.0f; rippleWeight = 0.0f;
    } else if (cloud_regime == 3) {
        cirrusWeight = 0.12f; veilWeight = 1.0f; rippleWeight = 0.04f;
    } else if (cloud_regime == 4) {
        cirrusWeight = 0.08f; veilWeight = 0.08f; rippleWeight = 1.0f;
    } else if (cloud_regime == 7) {
        cirrusWeight = 0.10f; veilWeight = 0.72f; rippleWeight = 0.04f;
    } else if (cloud_regime == 11) {
        // A cumulonimbus anvil reads as a broad ice shield. Long fibratus
        // strands are kept faint so they cannot turn into ruler-straight lines.
        cirrusWeight = 0.06f; veilWeight = 0.68f; rippleWeight = 0.02f;
    }
    float highCloud = cirrus * cirrusWeight
                    + cirrostratus * veilWeight
                    + cirrocumulus * rippleWeight;
    return saturate(highCloud * amount)
         * smoothstep(0.035f, 0.18f, rayDirection.y) * 0.45f;
}

)",
R"(
// Rain is a sparse world-space participating medium beneath the cloud base,
// rather than a screen-space overlay. Its attachment mask follows the same
// advected weather and body fields as the cloud above, so optically thick,
// moisture-rich regions produce the strongest precipitation naturally.
float rainSourceHeight() {
    float lowDeckPresent = step(0.15f, cloudDeckProfile().x);
    float lowerCloudInterior = cloudLayerBase()
        + max(cloud_radius * 0.42f, 0.55f);
    return lerp(middleLayerBase(), lowerCloudInterior, lowDeckPresent);
}

float sampleRainPotential(float2 sourceXZ, float sourceHeight) {
    float lowSource = step(0.15f, cloudDeckProfile().x);
    float sourceHeight01 = lerp(0.58f, 0.12f, lowSource);
    float3 sourcePoint = float3(sourceXZ.x, sourceHeight + 0.18f, sourceXZ.y);

    float weatherScale = 0.0125f
        / max(noise_scale * 0.35f + 0.65f, 0.25f);
    float3 weatherWind = animatedCloudOffset(sourcePoint, sourceHeight01, 0.72f);
    float2 weatherUv = (sourceXZ - weatherWind.xz) * weatherScale;
    float4 weather = CloudNoise.SampleLevel(
        CloudNoiseSampler, float3(weatherUv.x, 0.173f, weatherUv.y), 0.0f);

    // This local body sample is what keeps rain under individual dark cloud
    // masses instead of letting a low-frequency weather map rain everywhere.
    float frequency = max(noise_scale / max(billow_size, 0.25f), 0.05f);
    float3 bodyWind = animatedCloudOffset(sourcePoint, sourceHeight01, 1.0f);
    float3 lowUv = (sourcePoint - bodyWind)
        * float3(0.042f, 0.22f, 0.042f) * frequency
        + float3(0.18f, 0.11f, 0.29f);
    float3 middleUv = (sourcePoint - bodyWind)
        * float3(0.038f, 0.31f, 0.038f) * max(noise_scale, 0.05f) * 0.62f
        + float3(0.63f, 0.27f, 0.11f);
    float4 localBody = CloudNoise.SampleLevel(
        CloudNoiseSampler, lerp(middleUv, lowUv, lowSource), 0.0f);

    float regionalMoisture = weather.r * 0.44f + weather.a * 0.38f
                           + weather.g * 0.18f;
    float localMass = localBody.r * 0.58f + localBody.a * 0.30f
                    + localBody.g * 0.12f;
    float moisture = regionalMoisture * 0.55f + localMass * 0.45f;
    float normalizedMoisture = saturate((moisture - 0.10f) / 0.40f);
    // Low prevalence retains only the moist peaks; high prevalence smoothly
    // fills the same weather field instead of changing to an unrelated mask.
    float potential = pow(normalizedMoisture,
                          lerp(3.6f, 0.70f, saturate(rain_prevalence)))
                    * lerp(2.35f, 1.25f, saturate(rain_prevalence));
    float opticalThickness = saturate((cloud_density - 0.45f) * 0.34f)
                           * saturate(max(cloudDeckProfile().x,
                                          cloudDeckProfile().y * 0.88f));
    return potential * opticalThickness * max(rain_regime_factor, 0.0f);
}

float sampleRainDensity(float3 p, float sourceHeight, float anchorPotential) {
    if (p.y <= 0.02f || p.y >= sourceHeight) return 0.0f;
    float fall01 = saturate((sourceHeight - p.y) / max(sourceHeight - 0.02f, EPSILON));

    float2 direction = windDirectionXZ();
    float2 crossWind = float2(-direction.y, direction.x);
    float fallDistance = sourceHeight - p.y;
    float gust = smoothTriangle(dot(p.xz, float2(0.041f, 0.029f))
                              + time * 0.071f * max(rain_fall_speed, 0.05f));
    float2 fallDrift = direction * fallDistance * (0.13f + wind_speed * 0.16f)
                     + crossWind * gust * fallDistance * wind_turbulence * 0.055f;
    float2 sourceXZ = p.xz - fallDrift;
    // The cloud-base attachment is cached once per view ray. The falling
    // texture below still changes with depth, but the parent shower stays
    // coherent and avoids two weather-volume fetches at every rain step.
    float sourcePotential = anchorPotential;
    if (sourcePotential <= 0.001f) return 0.0f;

    float3 rainWind = animatedCloudOffset(p, saturate(p.y / sourceHeight), 0.92f);
    float2 advectedXZ = sourceXZ - rainWind.xz;
    float fallingCoordinate = p.y + time * max(rain_fall_speed, 0.0f) * 0.72f;
    float shaftFrequency = lerp(0.022f, 0.065f, saturate(rain_shaft_detail));
    float4 curtainNoise = CloudNoise.SampleLevel(
        CloudNoiseSampler,
        float3(advectedXZ.x * shaftFrequency,
               fallingCoordinate * 0.040f,
               advectedXZ.y * shaftFrequency) + float3(0.17f, 0.31f, 0.59f),
        0.0f);
    float4 filamentNoise = CloudNoise.SampleLevel(
        CloudNoiseSampler,
        float3(advectedXZ.x * shaftFrequency * 3.15f,
               fallingCoordinate * 0.085f,
               advectedXZ.y * shaftFrequency * 3.15f) + float3(0.61f, 0.13f, 0.37f),
        0.0f);
    float curtain = smoothstep(0.32f, 0.70f,
        curtainNoise.a * 0.54f + curtainNoise.r * 0.46f);
    float filaments = smoothstep(0.38f, 0.73f,
        filamentNoise.g * 0.56f + filamentNoise.r * 0.44f);
    float shaftStructure = curtain
        * lerp(0.20f + filaments * 1.35f, 0.62f + filaments * 0.48f,
               saturate(rain_mist));

    // Virga remains attached to the base but fades before the virtual ground.
    // At zero evaporation the fade point lies below the rendered atmosphere.
    float evaporationEnd = lerp(1.16f, 0.24f, saturate(rain_evaporation));
    float survival = 1.0f - smoothstep(
        max(evaporationEnd - 0.19f, 0.0f), evaporationEnd, fall01);
    float attachment = smoothstep(0.015f, 0.095f, fall01);
    float pulse = 0.78f + 0.22f * (smoothTriangle(
        time * 0.11f * max(rain_fall_speed, 0.05f)
      + dot(sourceXZ, float2(0.036f, 0.052f))) * 0.5f + 0.5f);
    pulse = lerp(pulse, 1.0f, saturate(rain_prevalence));
    return sourcePotential * shaftStructure * attachment * survival
         * pulse * max(rain_amount, 0.0f) * 3.4f;
}

bool rayRainBounds(float3 ro, float3 rd, out float nearT, out float farT) {
    if (rain_enabled == 0 || rain_amount <= 0.001f
        || rain_regime_factor <= 0.001f || rd.y <= 0.008f) {
        nearT = 0.0f;
        farT = 0.0f;
        return false;
    }
    float sourceHeight = rainSourceHeight();
    nearT = max((0.02f - ro.y) / rd.y, 0.0f);
    farT = min((sourceHeight - ro.y) / rd.y, 62.0f);
    return farT > nearT + 0.01f;
}

float4 raymarchRain(float3 rayOrigin, float3 rayDirection,
                    float3 sunDirection, float jitter) {
    float nearT, farT;
    if (!rayRainBounds(rayOrigin, rayDirection, nearT, farT)) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    int rainSteps = max(6, min(12, march_steps / 10));
    float stepLength = (farT - nearT) / (float)rainSteps;
    float depth = nearT + stepLength * jitter;
    float transmittance = 1.0f;
    float3 radiance = 0.0f;
    float sourceHeight = rainSourceHeight();
    float sourceT = (sourceHeight - rayOrigin.y) / max(rayDirection.y, 0.008f);
    float2 anchorXZ = (rayOrigin + rayDirection * sourceT).xz;
    float anchorPotential = sampleRainPotential(anchorXZ, sourceHeight);
    if (anchorPotential <= 0.001f) {
        return float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    float sunForward = pow(saturate(dot(rayDirection, sunDirection)), 6.0f);
    float daylight = smoothstep(-0.12f, 0.045f, sunDirection.y);
    float moonFill = moon_enabled != 0
        ? min(max(moon_brightness, 0.0f), 2.0f) * 0.12f : 0.0f;

    [loop]
    for (int i = 0; i < rainSteps; ++i) {
        if (depth > farT || transmittance < 0.015f) break;
        float3 p = rayOrigin + rayDirection * depth;
        float density = sampleRainDensity(p, sourceHeight, anchorPotential);
        if (density > 0.001f) {
            float extinction = density * stepLength
                * lerp(0.43f, 0.67f, saturate(rain_mist));
            float sampleAlpha = 1.0f - exp(-extinction);
            float3 rainLight = lerp(
                float3(0.29f, 0.35f, 0.41f),
                float3(0.67f, 0.71f, 0.75f),
                saturate(0.16f + sunForward * 0.72f + rain_mist * 0.12f));
            rainLight = lerp(
                float3(0.025f, 0.040f, 0.080f)
                    * (0.35f + max(night_brightness, 0.0f) + moonFill),
                rainLight, daylight);
            float aerial = exp(-depth * 0.018f);
            float3 rainAerial = lerp(
                float3(0.018f, 0.030f, 0.062f),
                float3(0.59f, 0.65f, 0.70f), daylight);
            rainLight = lerp(rainAerial, rainLight, aerial);
            radiance += transmittance * sampleAlpha * rainLight;
            transmittance *= exp(-extinction);
        }
        depth += stepLength;
    }
    return float4(radiance, transmittance);
}

)",
R"(
// Compact analytic atmosphere: solar elevation controls the day/night blend,
// while view elevation and angle from the sun approximate Rayleigh/Mie color
// variation. It follows the structure of analytic daylight models without
// their coefficient tables or a costly atmosphere raymarch.
float3 applySkyVibrancy(float3 color, float amount) {
    float luminance = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    return max(float3(0.0f, 0.0f, 0.0f),
               luminance.xxx + (color - luminance.xxx) * max(amount, 0.0f));
}

float2 starCubeProjection(float3 direction, out float faceIndex) {
    float3 axis = abs(direction);
    if (axis.x >= axis.y && axis.x >= axis.z) {
        faceIndex = direction.x >= 0.0f ? 0.0f : 1.0f;
        return direction.zy / max(axis.x, EPSILON);
    }
    if (axis.y >= axis.z) {
        faceIndex = direction.y >= 0.0f ? 2.0f : 3.0f;
        return direction.xz / max(axis.y, EPSILON);
    }
    faceIndex = direction.z >= 0.0f ? 4.0f : 5.0f;
    return direction.xy / max(axis.z, EPSILON);
}

float3 renderStars(float3 rayDirection, float3 sunDirection) {
    if (stars_enabled == 0 || star_amount <= 0.001f
        || star_brightness <= 0.001f) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float nightVisibility = 1.0f
        - smoothstep(-0.22f, -0.035f, sunDirection.y);
    float horizonVisibility = smoothstep(-0.035f, 0.20f, rayDirection.y);
    float visibility = nightVisibility * horizonVisibility;
    if (visibility <= 0.001f) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float faceIndex;
    float2 faceUv = starCubeProjection(rayDirection, faceIndex);
    float scale = lerp(150.0f, 390.0f, saturate(star_amount * 0.50f));
    float2 starPosition = (faceUv * 0.5f + 0.5f) * scale;
    float2 cell = floor(starPosition);
    float2 local = frac(starPosition) - 0.5f;
    float seed = hash12(cell + faceIndex * float2(137.2f, 311.7f));
    float density = min(max(star_amount, 0.0f) * 0.013f, 0.035f);
    float exists = step(1.0f - density, seed);
    float2 starCenter = float2(
        hash12(cell + faceIndex * 19.1f + 4.7f),
        hash12(cell + faceIndex * 31.3f + 8.2f)) - 0.5f;
    starCenter *= 0.62f;
    float starSeed = hash12(cell + faceIndex * 53.9f + 17.4f);
    float radius = lerp(0.035f, 0.13f, pow(starSeed, 7.0f));
    float pixelFootprint = max(scale / max((float)height, 1.0f) * 0.46f,
                               0.022f);
    float spark = 1.0f - smoothstep(
        radius, radius + pixelFootprint, length(local - starCenter));
    float twinklePhase = time * lerp(1.1f, 4.8f, starSeed)
                       + seed * (2.0f * PI);
    float twinkle = lerp(1.0f,
        0.72f + 0.28f * sin(twinklePhase), saturate(star_twinkle));
    float brightness = lerp(0.45f, 2.8f, pow(starSeed, 4.0f));
    float3 warmStar = float3(1.0f, 0.77f, 0.57f);
    float3 coolStar = float3(0.64f, 0.78f, 1.0f);
    float3 starColor = lerp(warmStar, coolStar,
        hash12(cell + faceIndex * 71.2f + 29.8f));
    return starColor * spark * exists * brightness * twinkle
         * max(star_brightness, 0.0f) * visibility;
}

float lunarEllipse(float2 p, float2 center, float2 radii,
                   float softness) {
    float distanceToBasin = length((p - center) / max(radii, 0.001f));
    return 1.0f - smoothstep(1.0f - softness,
                             1.0f + softness, distanceToBasin);
}

float lunarMaria(float2 p) {
    // Broad overlapping basalt basins approximate the recognizable near-side
    // geography instead of using uniform noise across the whole disc.
    float2 warpedPosition = p + float2(
        fbm2d4(p * 3.1f + 17.3f) - 0.5f,
        fbm2d4(p.yx * 3.4f + 41.8f) - 0.5f) * 0.075f;
    float maria = 0.0f;
    maria += lunarEllipse(warpedPosition, float2(-0.46f, 0.05f),
                          float2(0.32f, 0.53f), 0.22f) * 0.64f;
    maria += lunarEllipse(warpedPosition, float2(-0.22f, 0.39f),
                          float2(0.31f, 0.24f), 0.16f) * 0.78f;
    maria += lunarEllipse(warpedPosition, float2(0.16f, 0.37f),
                          float2(0.23f, 0.18f), 0.14f) * 0.72f;
    maria += lunarEllipse(warpedPosition, float2(0.31f, 0.10f),
                          float2(0.25f, 0.20f), 0.16f) * 0.74f;
    maria += lunarEllipse(warpedPosition, float2(0.44f, -0.20f),
                          float2(0.20f, 0.25f), 0.16f) * 0.62f;
    maria += lunarEllipse(warpedPosition, float2(-0.13f, -0.27f),
                          float2(0.27f, 0.18f), 0.18f) * 0.55f;
    maria += lunarEllipse(warpedPosition, float2(0.59f, 0.31f),
                          float2(0.13f, 0.16f), 0.12f) * 0.70f;
    float basinVariation = lerp(0.78f, 1.12f,
        fbm2d4(warpedPosition * 5.2f + 8.6f));
    return smoothstep(0.05f, 0.92f, saturate(maria * basinVariation));
}

float lunarCraterHeight(float2 p, float2 center, float radius,
                        float depth) {
    float normalizedDistance = length(p - center) / max(radius, 0.001f);
    float bowl = -depth * (1.0f - normalizedDistance * normalizedDistance)
               * (1.0f - smoothstep(0.68f, 1.0f, normalizedDistance));
    float rimDistance = (normalizedDistance - 1.0f) / 0.105f;
    float rim = depth * 0.48f * exp(-rimDistance * rimDistance);
    return (normalizedDistance < 1.42f) ? bowl + rim : 0.0f;
}

float lunarCellCraterHeight(float2 p, float scale, float seedOffset) {
    float2 gridPosition = (p * 0.5f + 0.5f) * scale;
    float2 cell = floor(gridPosition);
    float2 local = frac(gridPosition) - 0.5f;
    float2 center = float2(hash12(cell + seedOffset),
                           hash12(cell + seedOffset + 19.7f)) - 0.5f;
    center *= 0.66f;
    float probability = hash12(cell + seedOffset + 73.1f);
    float radius = lerp(0.075f, 0.24f,
                        hash12(cell + seedOffset + 43.1f));
    float crater = lunarCraterHeight(local, center, radius, 0.011f);
    return crater * step(0.68f, probability);
}

float lunarTerrainHeight(float2 p) {
    // A few persistent named-scale structures anchor the surface, while two
    // sparse cellular layers fill in small craters without an obvious grid.
    float heightField = 0.0f;
    heightField += lunarCraterHeight(p, float2(-0.28f, 0.03f), 0.105f, 0.018f);
    heightField += lunarCraterHeight(p, float2(0.18f, -0.55f), 0.125f, 0.022f);
    heightField += lunarCraterHeight(p, float2(0.50f, -0.05f), 0.080f, 0.014f);
    heightField += lunarCraterHeight(p, float2(-0.53f, 0.33f), 0.074f, 0.012f);
    heightField += lunarCellCraterHeight(p, 9.0f, 11.2f);
    heightField += lunarCellCraterHeight(p, 19.0f, 47.6f) * 0.42f;
    return heightField;
}

float lunarEjecta(float2 p, float2 center, float radius, float seed) {
    float2 offset = p - center;
    float distanceFromCrater = length(offset);
    float angle = atan2(offset.y, offset.x);
    float spokes = pow(saturate(0.50f + 0.27f * sin(angle * 11.0f + seed)
                              + 0.18f * sin(angle * 23.0f - seed)), 11.0f);
    float radialFade = smoothstep(radius, radius * 1.45f, distanceFromCrater)
                     * (1.0f - smoothstep(radius * 1.45f,
                                          radius * 6.2f, distanceFromCrater));
    return spokes * radialFade;
}

)",
R"(
float3 moonDirectionFromControls() {
    float azimuth = radians(moon_azimuth + time * moon_drift_speed);
    float elevation = radians(clamp(moon_elevation, -89.5f, 89.5f));
    float cosElevation = cos(elevation);
    return normalize(float3(
        sin(azimuth) * cosElevation,
        sin(elevation),
        cos(azimuth) * cosElevation));
}

float3 compositeMoon(float3 sky, float3 rayDirection,
                     float3 sunDirection) {
    if (moon_enabled == 0 || moon_size <= 0.001f
        || moon_brightness <= 0.001f) {
        return sky;
    }

    float3 moonDirection = moonDirectionFromControls();
    float centerFacing = dot(rayDirection, moonDirection);
    if (centerFacing <= 0.0f) return sky;

    float azimuth = radians(moon_azimuth + time * moon_drift_speed);
    float3 moonRight = float3(cos(azimuth), 0.0f, -sin(azimuth));
    float3 moonUp = normalize(cross(moonDirection, moonRight));
    float angularRadius = radians(0.26f * max(moon_size, 0.05f));
    float tangentRadius = max(tan(angularRadius), 0.0001f);
    float2 discPosition = float2(
        dot(rayDirection, moonRight), dot(rayDirection, moonUp))
        / max(tangentRadius * centerFacing, 0.0001f);
    float discRadius = length(discPosition);
    float pixelFootprint = max(
        2.0f / (max((float)height, 1.0f) * tangentRadius), 0.006f);

    // A restrained atmospheric aureole is evaluated outside the lunar disc.
    // Its strength follows illuminated fraction, so a crescent never blooms
    // like a full moon or turns into a giant soft white spot.
    float phaseAngle = saturate(moon_phase) * (2.0f * PI);
    float illuminatedFraction = 0.5f * (1.0f - cos(phaseAngle));
    float glowAmount = saturate(moon_glow * 0.5f);
    float haloDistance = max(discRadius - 1.0f, 0.0f);
    float halo = exp(-haloDistance * lerp(18.0f, 5.0f, glowAmount))
               * step(1.0f, discRadius)
               * (1.0f - smoothstep(2.8f, 6.0f, discRadius));
    float dayFactor = smoothstep(-0.12f, 0.05f, sunDirection.y);
    float3 haloColor = lerp(float3(0.38f, 0.43f, 0.54f),
                            float3(0.72f, 0.77f, 0.84f), dayFactor);
    sky += haloColor * halo * illuminatedFraction
         * glowAmount * max(moon_brightness, 0.0f) * 0.035f;

    if (discRadius > 1.0f + pixelFootprint) return sky;

    float normalZ = sqrt(saturate(1.0f - dot(discPosition, discPosition)));
    float3 lunarNormal = normalize(float3(discPosition, normalZ));
    // 0=new, .25=first quarter, .5=full, .75=last quarter, 1=new.
    // Lighting a visible sphere produces the correct curved terminator for
    // crescent, quarter, gibbous, and full phases.
    float3 phaseLight = normalize(float3(
        sin(phaseAngle), 0.08f * sin(phaseAngle * 0.5f), -cos(phaseAngle)));

    float detailAmount = max(moon_surface_detail, 0.0f);
    float terrainEpsilon = 0.0065f;
    float centerHeight = lunarTerrainHeight(discPosition);
    float terrainDx = (lunarTerrainHeight(
        discPosition + float2(terrainEpsilon, 0.0f)) - centerHeight)
        / terrainEpsilon;
    float terrainDy = (lunarTerrainHeight(
        discPosition + float2(0.0f, terrainEpsilon)) - centerHeight)
        / terrainEpsilon;
    float terminatorRelief = lerp(0.035f, 0.18f,
                                  1.0f - saturate(phaseLight.z));
    float3 terrainNormal = normalize(lunarNormal
        + float3(-terrainDx, -terrainDy, 0.0f)
          * detailAmount * terminatorRelief);

    // Lunar-Lambert reflectance blends Lambert diffuse with the
    // Lommel-Seeliger lunar term. This avoids the plastic sphere falloff of
    // pure Lambert shading while retaining deep relief at the terminator.
    float incidenceCosine = saturate(dot(terrainNormal, phaseLight));
    float emissionCosine = max(normalZ, 0.015f);
    float lommelSeeliger = (2.0f * incidenceCosine)
        / max(incidenceCosine + emissionCosine, 0.015f);
    float lunarReflectance = lerp(incidenceCosine,
                                  lommelSeeliger, 0.72f);
    float oppositionSurge = 1.0f
        + 0.24f * pow(saturate(phaseLight.z), 28.0f);
    lunarReflectance *= oppositionSurge;

    float4 lunarNoise = CloudNoise.SampleLevel(
        CloudNoiseSampler,
        lunarNormal * float3(0.41f, 0.33f, 0.41f)
        + float3(0.17f, 0.63f, 0.29f), 0.0f);
    float maria = lunarMaria(discPosition);
    float mottling = (lunarNoise.r * 0.56f + lunarNoise.a * 0.44f - 0.5f)
                   * 0.22f * detailAmount;
    float ejecta = lunarEjecta(discPosition, float2(0.18f, -0.55f),
                               0.125f, 2.1f)
                 + lunarEjecta(discPosition, float2(-0.28f, 0.03f),
                               0.105f, 4.7f) * 0.55f;
    float surfaceAlbedo = 0.68f - maria * 0.30f * detailAmount
                        + mottling + ejecta * 0.045f * detailAmount;
    surfaceAlbedo = clamp(surfaceAlbedo, 0.27f, 0.90f);

    // Earth is fullest from the lunar surface around our new/crescent Moon,
    // so earthshine fades as the directly illuminated fraction increases.
    float earthshinePhase = pow(saturate(1.0f - illuminatedFraction), 0.72f);
    float earthshine = max(moon_earthshine, 0.0f) * earthshinePhase
                     * (0.30f + emissionCosine * 0.70f);
    float directRadiance = lunarReflectance
                         * max(moon_brightness, 0.0f);
    float3 regolithColor = lerp(float3(0.51f, 0.50f, 0.47f),
                                float3(0.60f, 0.58f, 0.52f),
                                dayFactor * 0.32f);
    float3 earthshineColor = float3(0.31f, 0.37f, 0.48f);
    float3 moonColor = regolithColor * surfaceAlbedo * directRadiance
                     + earthshineColor * surfaceAlbedo * earthshine
                       * (1.0f - incidenceCosine);
    moonColor = max(moonColor, 0.0f);
    float discMask = 1.0f - smoothstep(
        1.0f - pixelFootprint, 1.0f + pixelFootprint, discRadius);
    float phaseVisibility = saturate(incidenceCosine * 7.0f
                          + earthshine * 10.0f);
    return lerp(sky, moonColor, discMask * phaseVisibility);
}

float3 renderAnalyticSky(float3 rayDirection, float3 sunDirection) {
    float solarElevation = sunDirection.y;
    float dayFactor = smoothstep(-0.12f, 0.045f, solarElevation);
    float highSun = smoothstep(0.035f, 0.58f, solarElevation);
    float viewHeight = saturate(rayDirection.y * 1.18f + 0.035f);
    float zenithBlend = pow(viewHeight, 0.58f);

    float3 nightHorizon = float3(0.025f, 0.030f, 0.070f);
    float3 nightZenith = float3(0.0025f, 0.006f, 0.023f);
    float nightScale = 0.28f + max(night_brightness, 0.0f) * 1.85f;
    float3 nightSky = lerp(nightHorizon, nightZenith, zenithBlend) * nightScale;

    float3 lowSunHorizon = float3(0.56f, 0.25f, 0.13f);
    float3 dayHorizon = float3(0.57f, 0.73f, 0.89f);
    float3 lowSunZenith = float3(0.055f, 0.075f, 0.22f);
    float3 dayZenith = float3(0.055f, 0.255f, 0.64f);
    float3 horizonColor = lerp(lowSunHorizon, dayHorizon, highSun);
    float3 zenithColor = lerp(lowSunZenith, dayZenith, highSun);
    float3 daylightSky = lerp(horizonColor, zenithColor, zenithBlend);
    float3 sky = lerp(nightSky, daylightSky, dayFactor);

    float2 viewHorizontal = normalize(rayDirection.xz + float2(0.0001f, 0.0f));
    float2 sunHorizontal = normalize(sunDirection.xz + float2(0.0001f, 0.0f));
    float towardSun = saturate(dot(viewHorizontal, sunHorizontal) * 0.5f + 0.5f);
    float twilightWidth = lerp(0.055f, 0.30f, saturate(twilight_range));
    float twilight = exp(-abs(solarElevation + 0.025f)
                     / max(twilightWidth, 0.02f))
                   * smoothstep(-0.38f, 0.12f, solarElevation);
    float horizonBand = exp(-abs(rayDirection.y) * 5.2f)
                      * smoothstep(-0.13f, 0.015f, rayDirection.y);
    float sunsetLobe = pow(towardSun, 2.8f);
    float3 orangeBand = float3(0.92f, 0.27f, 0.055f) * sunsetLobe * 0.22f;
    float3 magentaBand = float3(0.38f, 0.040f, 0.20f)
                       * pow(towardSun, 0.72f) * (1.0f - sunsetLobe * 0.38f);
    magentaBand *= 0.22f;
    float3 violetArch = float3(0.18f, 0.045f, 0.40f)
                      * smoothstep(0.0f, 0.58f, viewHeight)
                      * (1.0f - highSun) * 0.25f;
    sky += (orangeBand + magentaBand + violetArch)
         * twilight * horizonBand * max(twilight_intensity, 0.0f);

    float haze = exp(-abs(rayDirection.y) * 10.5f)
               * saturate(horizon_haze);
    float3 hazeColor = lerp(float3(0.085f, 0.075f, 0.13f),
                            lerp(lowSunHorizon, float3(0.77f, 0.81f, 0.86f),
                                 highSun), dayFactor);
    sky = lerp(sky, hazeColor, haze * lerp(0.12f, 0.42f, dayFactor));

    // When the camera tilts below the mathematical horizon, continue into a
    // subdued atmospheric ground haze instead of repeating the saturated
    // horizon color across the lower half of the frame.
    float belowHorizon = 1.0f - smoothstep(-0.13f, 0.012f, rayDirection.y);
    float lowerDepth = saturate(-rayDirection.y * 3.2f);
    float3 deepLowerSky = lerp(float3(0.010f, 0.013f, 0.028f),
                               float3(0.27f, 0.30f, 0.33f), dayFactor);
    float3 lowerSky = lerp(hazeColor * 0.58f, deepLowerSky, lowerDepth);
    sky = lerp(sky, lowerSky, belowHorizon);

    sky += renderStars(rayDirection, sunDirection);
    sky = applySkyVibrancy(sky, sky_vibrancy) * max(sky_exposure, 0.0f);

    float sunFacing = saturate(dot(rayDirection, sunDirection));
    float sunAngularDistance = sqrt(max(2.0f * (1.0f - sunFacing), 0.0f));
    float sunRadius = radians(0.266f * max(sun_disk_size, 0.05f));
    float angularPixel = max(1.7f / max((float)height, 1.0f), 0.00035f);
    float sunDisc = 1.0f - smoothstep(
        sunRadius, sunRadius + angularPixel, sunAngularDistance);
    float sunHalo = pow(sunFacing, 24.0f) * 0.11f
                  + pow(sunFacing, 220.0f) * 0.48f;
    float sunVisible = smoothstep(-0.035f, 0.012f, solarElevation);
    float3 lowSunColor = float3(1.0f, 0.52f, 0.24f);
    float3 sunColor = lerp(lowSunColor, float3(sun_r, sun_g, sun_b), highSun);
    sky += sunColor * (sunDisc * 1.65f + sunHalo)
         * max(sun_brightness, 0.0f) * sunVisible;

    return compositeMoon(sky, rayDirection, sunDirection);
}

)",
R"(
[numthreads(16, 16, 1)]
void CSMain(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= (uint)width || DTid.y >= (uint)height) return;
    uint2 pixel = DTid.xy;

    // UV coordinates centered at 0
    float2 uv = (float2(pixel) + 0.5f) / float2(width, height);
    uv -= 0.5f;
    uv.x *= (float)width / (float)height;

    // Camera Pitch is measured up from the horizon: 0 looks horizontally and
    // the 90-degree default looks straight into the zenith. The analytic basis
    // remains stable at 90 degrees, unlike a cross-product with world-up.
    float3 ro = float3(0.0f, 0.0f, 0.0f);
    float focalLength = max(camera_dist * 0.20f, 0.25f);
    float pitchRadians = radians(clamp(camera_pitch, -89.9f, 90.0f));
    float yawRadians = radians(camera_yaw);
    float cosPitch = cos(pitchRadians);
    float sinPitch = sin(pitchRadians);
    float cosYaw = cos(yawRadians);
    float sinYaw = sin(yawRadians);
    float3 cameraForward = float3(
        sinYaw * cosPitch,
        sinPitch,
        cosYaw * cosPitch);
    float3 cameraRight = float3(cosYaw, 0.0f, -sinYaw);
    float3 cameraUp = normalize(cross(cameraForward, cameraRight));
    float3 rd = normalize(
        cameraForward * focalLength
      + cameraRight * (uv.x * 1.04f)
      + cameraUp * (-uv.y * 0.92f));

    float3 sunDirection = normalize(float3(sun_x, sun_y, sun_z));
    float3 sky = renderAnalyticSky(rd, sunDirection);
    float cirrus = sampleCirrus(ro, rd);
    float daylight = smoothstep(-0.12f, 0.045f, sunDirection.y);
    float3 cirrusColor = lerp(
        float3(0.035f, 0.050f, 0.095f)
            * (0.45f + max(night_brightness, 0.0f)),
        float3(0.94f, 0.96f, 0.99f), daylight);
    sky = lerp(sky, cirrusColor, cirrus);

    // Keep the tiny source-layer contribution out of the GPU upload path. The
    // CPU bridge adds input.rgb * 5% * final transmittance during its existing
    // readback loop, exactly where it already restores the AE layer alpha.
    float3 background = sky * 0.95f;
    // AE has no temporal accumulation pass to hide a blue-noise depth phase.
    // A deterministic midpoint integral at full pixel resolution preserves
    // fine erosion without baking a stipple/checker pattern into every frame.
    float jitter = 0.5f;
    float4 lowCloud = raymarch(ro, rd, sunDirection, jitter);
    float4 middleCloud = raymarchMiddle(ro, rd, sunDirection, jitter);
    // From a ground observer the low deck is always in front of the middle
    // deck. Compose their radiance/transmittance before placing them over the
    // high ice layer and atmospheric background.
    float4 cloud = float4(
        lowCloud.rgb + lowCloud.a * middleCloud.rgb,
        lowCloud.a * middleCloud.a);
    float3 color = cloud.rgb + background * cloud.a;
    float4 rain = raymarchRain(ro, rd, sunDirection, jitter);
    // Precipitation occupies the sub-cloud atmosphere nearest the observer.
    color = rain.rgb + color * rain.a;

    // The AE 8-bit transfer below expects display-referred values already.
    color = saturate(color);

    // Preserve cloud/rain transmittance for the edge-aware resolve and for the
    // exact source-layer composite performed during readback.
    float4 result = float4(color, cloud.a * rain.a);

    OutputTexture[pixel] = result;
}

// Edge-aware spatial resolve for full-resolution interleaved depth samples.
// Transmittance prevents the filter from crossing cloud silhouettes.
[numthreads(16, 16, 1)]
void CSSharpen(uint3 DTid : SV_DispatchThreadID) {
    if (DTid.x >= (uint)width || DTid.y >= (uint)height) return;

    int2 p = int2(DTid.xy);
    int2 maximum = int2(width - 1, height - 1);
    float4 center = OutputTexture[p];

    float centerLuma = dot(center.rgb, float3(0.299f, 0.587f, 0.114f));
    float3 accumulated = center.rgb * 1.35f;
    float accumulatedWeight = 1.35f;

    [unroll]
    for (int oy = -1; oy <= 1; ++oy) {
        [unroll]
        for (int ox = -1; ox <= 1; ++ox) {
            if (ox == 0 && oy == 0) continue;
            int2 q = clamp(p + int2(ox, oy), int2(0, 0), maximum);
            float4 neighbor = OutputTexture[q];
            float neighborLuma = dot(neighbor.rgb, float3(0.299f, 0.587f, 0.114f));
            float spatialWeight = (ox == 0 || oy == 0) ? 1.0f : 0.72f;
            float transmittanceWeight = exp(-abs(neighbor.a - center.a) * 8.0f);
            float luminanceWeight = exp(-abs(neighborLuma - centerLuma) * 5.0f);
            float weight = spatialWeight * transmittanceWeight * luminanceWeight;
            accumulated += neighbor.rgb * weight;
            accumulatedWeight += weight;
        }
    }

    float3 filtered = accumulated / max(accumulatedWeight, EPSILON);
    // Full-resolution midpoint sampling has no checker noise to hide. Use the
    // bilateral neighborhood as a restrained unsharp mask so sub-billows read
    // clearly at 1080p/4K while transmittance gating prevents sky-edge halos.
    float volumetricOpacity = saturate((1.0f - center.a) * 8.0f);
    float3 resolved = center.rgb
                    + (center.rgb - filtered) * (0.42f * volumetricOpacity);
    InputTexture[p] = float4(saturate(resolved), center.a);
}
)" };

static std::string JoinShaderCode() {
    std::string shader;
    for (const char* chunk : g_shaderCodeChunks) {
        shader += chunk;
    }
    return shader;
}

static const std::string g_shaderCode = JoinShaderCode();

// GPU constant buffer structure (must match HLSL)
struct GPUCloudParams {
    float cloud_radius;
    float cloud_density;
    float absorption;
    float scattering;
    int march_steps;
    int light_steps;
    float sun_x, sun_y, sun_z;
    float sun_r, sun_g, sun_b;
    float wind_speed;
    float noise_scale;
    int detail;
    float camera_dist;
    float time;
    int width;
    int height;
    float camera_pitch;
    float camera_yaw;
    float wind_direction_x;
    float wind_direction_z;
    float wind_shear;
    float wind_turbulence;
    float cloud_coverage;
    float cloud_type;
    float erosion_strength;
    float billow_size;
    float mid_level_amount;
    float high_level_amount;
    float storm_development;
    int cloud_regime;
    float low_level_amount;
    float mid_cellular_style;
    float low_coverage_offset;
    float mid_coverage_offset;
    float tower_development;
    float tower_scale;
    float cauliflower_detail;
    float tower_position_x;
    float tower_distance;
    float tower_isolation;
    float tower_padding_0;
    int rain_enabled;
    float rain_amount;
    float rain_prevalence;
    float rain_shaft_detail;
    float rain_mist;
    float rain_evaporation;
    float rain_fall_speed;
    float rain_regime_factor;
    float sky_vibrancy;
    float sky_exposure;
    float twilight_intensity;
    float twilight_range;
    float horizon_haze;
    float sun_disk_size;
    float sun_brightness;
    float night_brightness;
    int stars_enabled;
    float star_amount;
    float star_brightness;
    float star_twinkle;
    int moon_enabled;
    float moon_azimuth;
    float moon_elevation;
    float moon_size;
    float moon_phase;
    float moon_brightness;
    float moon_glow;
    float moon_drift_speed;
    float moon_surface_detail;
    float moon_earthshine;
    float sky_padding_0;
    float sky_padding_1;
};

static_assert(sizeof(GPUCloudParams) == 304,
              "GPU cloud constant buffer must match the HLSL layout");

// Initialize DirectX 11 GPU resources
static bool InitializeGPU() {
    if (g_gpuInitialized) return true;
    if (g_gpuFailed) return false;

    // Create D3D11 device
    D3D_FEATURE_LEVEL featureLevel;
    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        nullptr, 0,
        D3D11_SDK_VERSION,
        &g_device,
        &featureLevel,
        &g_context
    );

    if (FAILED(hr)) {
        // WARP runs the exact same HLSL on Microsoft's software rasterizer,
        // preserving visual parity on systems without a usable hardware device.
        hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_WARP,
            nullptr,
            flags & ~D3D11_CREATE_DEVICE_DEBUG,
            nullptr, 0,
            D3D11_SDK_VERSION,
            &g_device,
            &featureLevel,
            &g_context
        );
        if (FAILED(hr)) {
            g_gpuFailed = true;
            return false;
        }
    }

    // Compile compute shader
    ID3DBlob* shaderBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;

    hr = D3DCompile(
        g_shaderCode.data(),
        g_shaderCode.size(),
        "CloudShader",
        nullptr,
        nullptr,
        "CSMain",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shaderBlob,
        &errorBlob
    );

    if (FAILED(hr)) {
        if (errorBlob) errorBlob->Release();
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }

    // Create compute shader
    hr = g_device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &g_computeShader
    );

    shaderBlob->Release();

    if (FAILED(hr)) {
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }

    // Compile the one-time procedural 3D noise bake from the same HLSL source.
    shaderBlob = nullptr;
    errorBlob = nullptr;
    hr = D3DCompile(
        g_shaderCode.data(),
        g_shaderCode.size(),
        "CloudNoiseShader",
        nullptr,
        nullptr,
        "CSNoise",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shaderBlob,
        &errorBlob
    );
    if (FAILED(hr)) {
        if (errorBlob) errorBlob->Release();
        if (g_computeShader) { g_computeShader->Release(); g_computeShader = nullptr; }
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }

    hr = g_device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &g_noiseShader
    );
    shaderBlob->Release();
    if (FAILED(hr)) {
        if (g_computeShader) { g_computeShader->Release(); g_computeShader = nullptr; }
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }

    shaderBlob = nullptr;
    errorBlob = nullptr;
    hr = D3DCompile(
        g_shaderCode.data(),
        g_shaderCode.size(),
        "CloudSharpenShader",
        nullptr,
        nullptr,
        "CSSharpen",
        "cs_5_0",
        D3DCOMPILE_OPTIMIZATION_LEVEL3,
        0,
        &shaderBlob,
        &errorBlob
    );
    if (FAILED(hr)) {
        if (errorBlob) errorBlob->Release();
        if (g_noiseShader) { g_noiseShader->Release(); g_noiseShader = nullptr; }
        if (g_computeShader) { g_computeShader->Release(); g_computeShader = nullptr; }
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }
    hr = g_device->CreateComputeShader(
        shaderBlob->GetBufferPointer(),
        shaderBlob->GetBufferSize(),
        nullptr,
        &g_sharpenShader
    );
    shaderBlob->Release();
    if (FAILED(hr)) {
        if (g_noiseShader) { g_noiseShader->Release(); g_noiseShader = nullptr; }
        if (g_computeShader) { g_computeShader->Release(); g_computeShader = nullptr; }
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }

    D3D11_TEXTURE3D_DESC noiseDesc = {};
    noiseDesc.Width = 128;
    noiseDesc.Height = 128;
    noiseDesc.Depth = 128;
    noiseDesc.MipLevels = 1;
    // Every baked channel is normalized to 0..1 and sampled with linear
    // filtering. RGBA8 preserves 256 density/erosion levels while halving the
    // persistent 128^3 volume from 16 MiB to 8 MiB.
    noiseDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    noiseDesc.Usage = D3D11_USAGE_DEFAULT;
    noiseDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    hr = g_device->CreateTexture3D(&noiseDesc, nullptr, &g_noiseTex);
    if (SUCCEEDED(hr)) hr = g_device->CreateShaderResourceView(g_noiseTex, nullptr, &g_noiseSRV);
    if (SUCCEEDED(hr)) hr = g_device->CreateUnorderedAccessView(g_noiseTex, nullptr, &g_noiseUAV);

    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    // The procedural bake is not periodic at its borders. Mirroring keeps
    // filtering continuous where world coordinates cross a texture boundary.
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_MIRROR;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_MIRROR;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_MIRROR;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    if (SUCCEEDED(hr)) hr = g_device->CreateSamplerState(&samplerDesc, &g_noiseSampler);
    if (FAILED(hr)) {
        if (g_noiseSampler) { g_noiseSampler->Release(); g_noiseSampler = nullptr; }
        if (g_noiseUAV) { g_noiseUAV->Release(); g_noiseUAV = nullptr; }
        if (g_noiseSRV) { g_noiseSRV->Release(); g_noiseSRV = nullptr; }
        if (g_noiseTex) { g_noiseTex->Release(); g_noiseTex = nullptr; }
        if (g_sharpenShader) { g_sharpenShader->Release(); g_sharpenShader = nullptr; }
        if (g_noiseShader) { g_noiseShader->Release(); g_noiseShader = nullptr; }
        if (g_computeShader) { g_computeShader->Release(); g_computeShader = nullptr; }
        if (g_device) { g_device->Release(); g_device = nullptr; }
        if (g_context) { g_context->Release(); g_context = nullptr; }
        g_gpuFailed = true;
        return false;
    }

    // Bake and unbind before the volume becomes an SRV for cloud rendering.
    g_context->CSSetShader(g_noiseShader, nullptr, 0);
    g_context->CSSetUnorderedAccessViews(2, 1, &g_noiseUAV, nullptr);
    g_context->Dispatch(32, 32, 32);
    ID3D11UnorderedAccessView* nullNoiseUAV = nullptr;
    g_context->CSSetUnorderedAccessViews(2, 1, &nullNoiseUAV, nullptr);
    g_context->CSSetShader(nullptr, nullptr, 0);

    // The bake shader and its UAV are one-shot construction resources. The
    // rendered volume remains alive through its SRV for all subsequent frames.
    g_noiseUAV->Release();
    g_noiseUAV = nullptr;
    g_noiseShader->Release();
    g_noiseShader = nullptr;

    g_gpuInitialized = true;
    return true;
}

// Helper to release cached GPU resources
static void ReleaseGPUResources() {
    if (g_inputUAV) { g_inputUAV->Release(); g_inputUAV = nullptr; }
    if (g_outputUAV) { g_outputUAV->Release(); g_outputUAV = nullptr; }
    if (g_inputTex) { g_inputTex->Release(); g_inputTex = nullptr; }
    if (g_outputTex) { g_outputTex->Release(); g_outputTex = nullptr; }
    if (g_stagingTex) { g_stagingTex->Release(); g_stagingTex = nullptr; }
    if (g_constantBuffer) { g_constantBuffer->Release(); g_constantBuffer = nullptr; }
    g_cachedWidth = 0;
    g_cachedHeight = 0;
}

// GPU rendering function with cached resources
static bool RenderOnGPU(
    PF_Pixel8* in_data, PF_Pixel8* out_data,
    int width, int height,
    int in_rowbytes, int out_rowbytes,
    GPUCloudParams* params
) {
    // Lock mutex for thread-safe GPU access
    std::lock_guard<std::mutex> lock(g_gpuMutex);

    if (!InitializeGPU()) return false;

    HRESULT hr;

    // Recreate resources if dimensions changed
    if (width != g_cachedWidth || height != g_cachedHeight) {
        ReleaseGPUResources();

        // Create constant buffer
        D3D11_BUFFER_DESC cbDesc = {};
        cbDesc.ByteWidth = sizeof(GPUCloudParams);
        cbDesc.Usage = D3D11_USAGE_DYNAMIC;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        hr = g_device->CreateBuffer(&cbDesc, nullptr, &g_constantBuffer);
        if (FAILED(hr)) return false;

        // Create textures
        D3D11_TEXTURE2D_DESC texDesc = {};
        texDesc.Width = width;
        texDesc.Height = height;
        texDesc.MipLevels = 1;
        texDesc.ArraySize = 1;
        // The AE bridge is PF_Pixel8 in both directions. RGBA8 UNORM cuts each
        // cached 2D resource from 16 to 4 bytes/pixel; all raymarch math still
        // executes in float registers and only the pass boundary is quantized.
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        texDesc.Usage = D3D11_USAGE_DEFAULT;
        texDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;

        hr = g_device->CreateTexture2D(&texDesc, nullptr, &g_inputTex);
        if (FAILED(hr)) { ReleaseGPUResources(); return false; }

        hr = g_device->CreateTexture2D(&texDesc, nullptr, &g_outputTex);
        if (FAILED(hr)) { ReleaseGPUResources(); return false; }

        // Create UAVs
        hr = g_device->CreateUnorderedAccessView(g_inputTex, nullptr, &g_inputUAV);
        if (FAILED(hr)) { ReleaseGPUResources(); return false; }

        hr = g_device->CreateUnorderedAccessView(g_outputTex, nullptr, &g_outputUAV);
        if (FAILED(hr)) { ReleaseGPUResources(); return false; }

        // Create staging texture for readback
        D3D11_TEXTURE2D_DESC stagingDesc = texDesc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.BindFlags = 0;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        hr = g_device->CreateTexture2D(&stagingDesc, nullptr, &g_stagingTex);
        if (FAILED(hr)) { ReleaseGPUResources(); return false; }

        g_cachedWidth = width;
        g_cachedHeight = height;
    }

    // Update constant buffer
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = g_context->Map(g_constantBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr)) return false;
    memcpy(mapped.pData, params, sizeof(GPUCloudParams));
    g_context->Unmap(g_constantBuffer, 0);

    // Set up and dispatch compute shader
    g_context->CSSetShader(g_computeShader, nullptr, 0);
    g_context->CSSetConstantBuffers(0, 1, &g_constantBuffer);
    g_context->CSSetShaderResources(0, 1, &g_noiseSRV);
    g_context->CSSetSamplers(0, 1, &g_noiseSampler);
    ID3D11UnorderedAccessView* uavs[] = { g_inputUAV, g_outputUAV };
    g_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);

    UINT groupsX = (width + 15) / 16;
    UINT groupsY = (height + 15) / 16;
    g_context->Dispatch(groupsX, groupsY, 1);

    // Unbind between passes to establish the UAV dependency, then resolve the
    // cloud into the no-longer-needed input texture.
    ID3D11UnorderedAccessView* nullUAVs[] = { nullptr, nullptr };
    g_context->CSSetUnorderedAccessViews(0, 2, nullUAVs, nullptr);
    ID3D11ShaderResourceView* nullSRV = nullptr;
    g_context->CSSetShaderResources(0, 1, &nullSRV);

    g_context->CSSetShader(g_sharpenShader, nullptr, 0);
    g_context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
    g_context->Dispatch(groupsX, groupsY, 1);
    g_context->CSSetUnorderedAccessViews(0, 2, nullUAVs, nullptr);
    g_context->CSSetShader(nullptr, nullptr, 0);
    ID3D11Buffer* nullConstantBuffer = nullptr;
    ID3D11SamplerState* nullSampler = nullptr;
    g_context->CSSetConstantBuffers(0, 1, &nullConstantBuffer);
    g_context->CSSetSamplers(0, 1, &nullSampler);

    // Read back results
    g_context->CopyResource(g_stagingTex, g_inputTex);

    hr = g_context->Map(g_stagingTex, 0, D3D11_MAP_READ, 0, &mapped);
    if (SUCCEEDED(hr)) {
        const unsigned char* outputBytes =
            static_cast<const unsigned char*>(mapped.pData);

        for (int y = 0; y < height; y++) {
            PF_Pixel8* outRow = (PF_Pixel8*)((char*)out_data + y * out_rowbytes);
            const PF_Pixel8* inRow =
                (const PF_Pixel8*)((const char*)in_data + y * in_rowbytes);
            const unsigned char* sourceRow = outputBytes + y * mapped.RowPitch;
            for (int x = 0; x < width; x++) {
                int srcIdx = x * 4;
                float sourceWeight = sourceRow[srcIdx + 3] * (0.05f / 255.0f);
                int red = sourceRow[srcIdx + 0]
                        + (int)(inRow[x].red * sourceWeight + 0.5f);
                int green = sourceRow[srcIdx + 1]
                          + (int)(inRow[x].green * sourceWeight + 0.5f);
                int blue = sourceRow[srcIdx + 2]
                         + (int)(inRow[x].blue * sourceWeight + 0.5f);
                outRow[x].red = (A_u_char)((std::min)(red, 255));
                outRow[x].green = (A_u_char)((std::min)(green, 255));
                outRow[x].blue = (A_u_char)((std::min)(blue, 255));
                outRow[x].alpha = inRow[x].alpha;
            }
        }
        g_context->Unmap(g_stagingTex, 0);
    }

    return SUCCEEDED(hr);
}

#define NAME "VolumetricCloudShader"
#define DESCRIPTION "Volumetric weather with explosive towers, day/night sky, stars, and phased moon"
#define MAJOR_VERSION 1
#define MINOR_VERSION 12
#define BUG_VERSION 0
#define STAGE_VERSION PF_Stage_DEVELOP
#define BUILD_VERSION 1

#define PI 3.14159265359f

enum {
    INPUT_LAYER = 0,
    CLOUD_VOLUME_GROUP_START,
    CLOUD_RADIUS_PARAM,
    CLOUD_DENSITY_PARAM,
    ABSORPTION_PARAM,
    SCATTERING_PARAM,
    MARCH_STEPS_PARAM,
    LIGHT_STEPS_PARAM,
    CLOUD_VOLUME_GROUP_END,
    MANUAL_SUN_GROUP_START,
    SUN_X_PARAM,
    SUN_Y_PARAM,
    SUN_Z_PARAM,
    SUN_COLOR_PARAM,
    MANUAL_SUN_GROUP_END,
    CAMERA_GROUP_START,
    CAMERA_DIST_PARAM,
    CAMERA_PITCH_PARAM,
    CAMERA_YAW_PARAM,
    CAMERA_GROUP_END,
    CLOUD_SHAPE_GROUP_START,
    WIND_SPEED_PARAM,
    NOISE_SCALE_PARAM,
    DETAIL_PARAM,
    CLOUD_COVERAGE_PARAM,
    CLOUD_TYPE_PARAM,
    EROSION_STRENGTH_PARAM,
    BILLOW_SIZE_PARAM,
    WIND_DIRECTION_PARAM,
    WIND_SHEAR_PARAM,
    WIND_TURBULENCE_PARAM,
    CLOUD_SHAPE_GROUP_END,
    CLOUD_LAYERS_GROUP_START,
    CLOUD_REGIME_PARAM,
    MID_LEVEL_AMOUNT_PARAM,
    HIGH_LEVEL_AMOUNT_PARAM,
    STORM_DEVELOPMENT_PARAM,
    CLOUD_LAYERS_GROUP_END,
    HERO_TOWER_GROUP_START,
    TOWER_DEVELOPMENT_PARAM,
    TOWER_SCALE_PARAM,
    CAULIFLOWER_DETAIL_PARAM,
    TOWER_POSITION_X_PARAM,
    TOWER_DISTANCE_PARAM,
    TOWER_ISOLATION_PARAM,
    HERO_TOWER_GROUP_END,
    RAIN_GROUP_START,
    RAIN_ENABLED_PARAM,
    RAIN_AMOUNT_PARAM,
    RAIN_PREVALENCE_PARAM,
    RAIN_SHAFT_DETAIL_PARAM,
    RAIN_MIST_PARAM,
    RAIN_EVAPORATION_PARAM,
    RAIN_FALL_SPEED_PARAM,
    RAIN_GROUP_END,
    DAY_SKY_GROUP_START,
    DAY_CYCLE_ENABLED_PARAM,
    TIME_OF_DAY_PARAM,
    DAY_CYCLE_SPEED_PARAM,
    SUN_PATH_ROTATION_PARAM,
    SUN_PEAK_ELEVATION_PARAM,
    SKY_VIBRANCY_PARAM,
    SKY_EXPOSURE_PARAM,
    TWILIGHT_INTENSITY_PARAM,
    TWILIGHT_RANGE_PARAM,
    HORIZON_HAZE_PARAM,
    SUN_DISK_SIZE_PARAM,
    SUN_BRIGHTNESS_PARAM,
    NIGHT_BRIGHTNESS_PARAM,
    DAY_SKY_GROUP_END,
    STARS_GROUP_START,
    STARS_ENABLED_PARAM,
    STAR_AMOUNT_PARAM,
    STAR_BRIGHTNESS_PARAM,
    STAR_TWINKLE_PARAM,
    STARS_GROUP_END,
    MOON_GROUP_START,
    MOON_ENABLED_PARAM,
    MOON_AZIMUTH_PARAM,
    MOON_ELEVATION_PARAM,
    MOON_DRIFT_SPEED_PARAM,
    MOON_SIZE_PARAM,
    MOON_PHASE_PARAM,
    MOON_BRIGHTNESS_PARAM,
    MOON_GLOW_PARAM,
    MOON_SURFACE_DETAIL_PARAM,
    MOON_EARTHSHINE_PARAM,
    MOON_GROUP_END,
    NUM_PARAMS
};

// Parameter array indices above are free to follow the organized UI. These
// disk IDs deliberately retain the original v1.11 identities so moving a
// control into a topic does not remap saved values and keyframes by position.
enum {
    CLOUD_RADIUS_DISK_ID = 1,
    CLOUD_DENSITY_DISK_ID = 2,
    ABSORPTION_DISK_ID = 3,
    SCATTERING_DISK_ID = 4,
    MARCH_STEPS_DISK_ID = 5,
    LIGHT_STEPS_DISK_ID = 6,
    SUN_X_DISK_ID = 7,
    SUN_Y_DISK_ID = 8,
    SUN_Z_DISK_ID = 9,
    SUN_COLOR_DISK_ID = 10,
    WIND_SPEED_DISK_ID = 11,
    NOISE_SCALE_DISK_ID = 12,
    DETAIL_DISK_ID = 13,
    CAMERA_DIST_DISK_ID = 14,
    CAMERA_PITCH_DISK_ID = 15,
    CAMERA_YAW_DISK_ID = 16,
    CLOUD_COVERAGE_DISK_ID = 17,
    CLOUD_TYPE_DISK_ID = 18,
    EROSION_STRENGTH_DISK_ID = 19,
    BILLOW_SIZE_DISK_ID = 20,
    WIND_DIRECTION_DISK_ID = 21,
    WIND_SHEAR_DISK_ID = 22,
    WIND_TURBULENCE_DISK_ID = 23,
    CLOUD_REGIME_DISK_ID = 24,
    MID_LEVEL_AMOUNT_DISK_ID = 25,
    HIGH_LEVEL_AMOUNT_DISK_ID = 26,
    STORM_DEVELOPMENT_DISK_ID = 27,
    TOWER_DEVELOPMENT_DISK_ID = 28,
    TOWER_SCALE_DISK_ID = 29,
    CAULIFLOWER_DETAIL_DISK_ID = 30,
    TOWER_POSITION_X_DISK_ID = 31,
    TOWER_DISTANCE_DISK_ID = 32,
    TOWER_ISOLATION_DISK_ID = 33,
    RAIN_ENABLED_DISK_ID = 34,
    RAIN_AMOUNT_DISK_ID = 35,
    RAIN_PREVALENCE_DISK_ID = 36,
    RAIN_SHAFT_DETAIL_DISK_ID = 37,
    RAIN_MIST_DISK_ID = 38,
    RAIN_EVAPORATION_DISK_ID = 39,
    RAIN_FALL_SPEED_DISK_ID = 40,
    DAY_CYCLE_ENABLED_DISK_ID = 41,
    TIME_OF_DAY_DISK_ID = 42,
    DAY_CYCLE_SPEED_DISK_ID = 43,
    SUN_PATH_ROTATION_DISK_ID = 44,
    SUN_PEAK_ELEVATION_DISK_ID = 45,
    SKY_VIBRANCY_DISK_ID = 46,
    SKY_EXPOSURE_DISK_ID = 47,
    TWILIGHT_INTENSITY_DISK_ID = 48,
    TWILIGHT_RANGE_DISK_ID = 49,
    HORIZON_HAZE_DISK_ID = 50,
    SUN_DISK_SIZE_DISK_ID = 51,
    SUN_BRIGHTNESS_DISK_ID = 52,
    NIGHT_BRIGHTNESS_DISK_ID = 53,
    STARS_ENABLED_DISK_ID = 54,
    STAR_AMOUNT_DISK_ID = 55,
    STAR_BRIGHTNESS_DISK_ID = 56,
    STAR_TWINKLE_DISK_ID = 57,
    MOON_ENABLED_DISK_ID = 58,
    MOON_AZIMUTH_DISK_ID = 59,
    MOON_ELEVATION_DISK_ID = 60,
    MOON_DRIFT_SPEED_DISK_ID = 61,
    MOON_SIZE_DISK_ID = 62,
    MOON_PHASE_DISK_ID = 63,
    MOON_BRIGHTNESS_DISK_ID = 64,
    MOON_GLOW_DISK_ID = 65,
    MOON_SURFACE_DETAIL_DISK_ID = 66,
    MOON_EARTHSHINE_DISK_ID = 67,

    CLOUD_VOLUME_GROUP_DISK_ID = 1001,
    CLOUD_VOLUME_GROUP_END_DISK_ID = 1002,
    MANUAL_SUN_GROUP_DISK_ID = 1003,
    MANUAL_SUN_GROUP_END_DISK_ID = 1004,
    CAMERA_GROUP_DISK_ID = 1005,
    CAMERA_GROUP_END_DISK_ID = 1006,
    CLOUD_SHAPE_GROUP_DISK_ID = 1007,
    CLOUD_SHAPE_GROUP_END_DISK_ID = 1008,
    CLOUD_LAYERS_GROUP_DISK_ID = 1009,
    CLOUD_LAYERS_GROUP_END_DISK_ID = 1010,
    HERO_TOWER_GROUP_DISK_ID = 1011,
    HERO_TOWER_GROUP_END_DISK_ID = 1012,
    RAIN_GROUP_DISK_ID = 1013,
    RAIN_GROUP_END_DISK_ID = 1014,
    DAY_SKY_GROUP_DISK_ID = 1015,
    DAY_SKY_GROUP_END_DISK_ID = 1016,
    STARS_GROUP_DISK_ID = 1017,
    STARS_GROUP_END_DISK_ID = 1018,
    MOON_GROUP_DISK_ID = 1019,
    MOON_GROUP_END_DISK_ID = 1020
};

typedef struct {
    float cloud_radius;
    float cloud_density;
    float absorption;
    float scattering;
    int march_steps;
    int light_steps;
    float sun_x, sun_y, sun_z;
    float sun_r, sun_g, sun_b;
    float wind_speed;
    float noise_scale;
    int detail;
    float camera_dist;
    float camera_pitch;
    float camera_yaw;
    float wind_direction;
    float wind_shear;
    float wind_turbulence;
    float cloud_coverage;
    float cloud_type;
    float erosion_strength;
    float billow_size;
    float mid_level_amount;
    float high_level_amount;
    float storm_development;
    int cloud_regime;
    int rain_enabled;
    float rain_amount;
    float rain_prevalence;
    float rain_shaft_detail;
    float rain_mist;
    float rain_evaporation;
    float rain_fall_speed;
    float time;
} CloudShaderParams;

// ============================================================================
// OPTIMIZED MATH UTILITIES
// ============================================================================

static inline float fract(float x) {
    return x - floorf(x);
}

static inline float mix(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline float clampf(float x, float minVal, float maxVal) {
    return x < minVal ? minVal : (x > maxVal ? maxVal : x);
}

static inline float smoothstepHost(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline float dot3(float x1, float y1, float z1, float x2, float y2, float z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
}

static inline float length3(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

static inline void normalize3(float *x, float *y, float *z) {
    float len = length3(*x, *y, *z);
    if (len > 0.0001f) {
        float inv = 1.0f / len;
        *x *= inv;
        *y *= inv;
        *z *= inv;
    }
}

// Ray-sphere intersection
static inline bool raySphereIntersect(float roX, float roY, float roZ,
                                       float rdX, float rdY, float rdZ,
                                       float radius) {
    float b = 2.0f * (roX * rdX + roY * rdY + roZ * rdZ);
    float c = roX * roX + roY * roY + roZ * roZ - radius * radius;
    return (b * b - 4.0f * c) >= 0.0f;
}

// ============================================================================
// OPTIMIZED NOISE (using fast hash)
// ============================================================================

// Smooth gradient noise with quintic interpolation for soft cloud edges
static inline void hash33(float x, float y, float z, float* ox, float* oy, float* oz) {
    float px = x * 127.1f + y * 311.7f + z * 74.7f;
    float py = x * 269.5f + y * 183.3f + z * 246.1f;
    float pz = x * 113.5f + y * 271.9f + z * 124.6f;
    *ox = fract(sinf(px) * 43758.5453f) * 2.0f - 1.0f;
    *oy = fract(sinf(py) * 43758.5453f) * 2.0f - 1.0f;
    *oz = fract(sinf(pz) * 43758.5453f) * 2.0f - 1.0f;
}

static float noise3d(float x, float y, float z) {
    float ix = floorf(x);
    float iy = floorf(y);
    float iz = floorf(z);
    float fx = x - ix;
    float fy = y - iy;
    float fz = z - iz;

    // Quintic interpolation for smoother results (less sharp edges)
    float ux = fx * fx * fx * (fx * (fx * 6.0f - 15.0f) + 10.0f);
    float uy = fy * fy * fy * (fy * (fy * 6.0f - 15.0f) + 10.0f);
    float uz = fz * fz * fz * (fz * (fz * 6.0f - 15.0f) + 10.0f);

    // Gradient noise
    float gax, gay, gaz, gbx, gby, gbz, gcx, gcy, gcz, gdx, gdy, gdz;
    float gex, gey, gez, gfx, gfy, gfz, ggx, ggy, ggz, ghx, ghy, ghz;

    hash33(ix, iy, iz, &gax, &gay, &gaz);
    hash33(ix+1, iy, iz, &gbx, &gby, &gbz);
    hash33(ix, iy+1, iz, &gcx, &gcy, &gcz);
    hash33(ix+1, iy+1, iz, &gdx, &gdy, &gdz);
    hash33(ix, iy, iz+1, &gex, &gey, &gez);
    hash33(ix+1, iy, iz+1, &gfx, &gfy, &gfz);
    hash33(ix, iy+1, iz+1, &ggx, &ggy, &ggz);
    hash33(ix+1, iy+1, iz+1, &ghx, &ghy, &ghz);

    float va = gax * fx + gay * fy + gaz * fz;
    float vb = gbx * (fx-1) + gby * fy + gbz * fz;
    float vc = gcx * fx + gcy * (fy-1) + gcz * fz;
    float vd = gdx * (fx-1) + gdy * (fy-1) + gdz * fz;
    float ve = gex * fx + gey * fy + gez * (fz-1);
    float vf = gfx * (fx-1) + gfy * fy + gfz * (fz-1);
    float vg = ggx * fx + ggy * (fy-1) + ggz * (fz-1);
    float vh = ghx * (fx-1) + ghy * (fy-1) + ghz * (fz-1);

    float x0 = mix(va, vb, ux);
    float x1 = mix(vc, vd, ux);
    float x2 = mix(ve, vf, ux);
    float x3 = mix(vg, vh, ux);

    return mix(mix(x0, x1, uy), mix(x2, x3, uy), uz);
}

// FBM with limited octaves - matches original GLSL
static float fbm(float x, float y, float z, int octaves, float time) {
    // Apply time-based wind movement
    x += time * 0.5f;
    y -= time * 0.1f;
    z -= time * 0.5f;

    float f = 0.0f;
    float scale = 0.45f;
    float factor = 2.02f;

    for (int i = 0; i < octaves; i++) {
        f += scale * noise3d(x, y, z);
        x *= factor;
        y *= factor;
        z *= factor;
        factor += 0.21f;
        scale *= 0.5f;
    }

    return f;
}

// Scene density
static inline float scene(float px, float py, float pz, CloudShaderParams *params, bool lowRes) {
    float dist = length3(px, py, pz) - params->cloud_radius;
    int oct = lowRes ? 2 : params->detail;
    float f = fbm(px * params->noise_scale, py * params->noise_scale,
                  pz * params->noise_scale, oct, params->time);
    return -dist + f;
}

// Beer's Law
static inline float BeersLaw(float dist, float absorption) {
    return expf(-dist * absorption);
}

// Henyey-Greenstein phase function
static inline float HenyeyGreenstein(float g, float mu) {
    float gg = g * g;
    float denom = 1.0f + gg - 2.0f * g * mu;
    if (denom < 0.001f) denom = 0.001f;
    return (0.25f / PI) * ((1.0f - gg) / powf(denom, 1.5f));
}

// Light marching - matches original GLSL
static float lightmarch(float px, float py, float pz, CloudShaderParams *params,
                        float sunDirX, float sunDirY, float sunDirZ) {
    float totalDensity = 0.0f;
    float marchSize = 0.03f;

    for (int step = 0; step < params->light_steps; step++) {
        // Original uses: position += sunDirection * marchSize * float(step)
        px += sunDirX * marchSize * (float)step;
        py += sunDirY * marchSize * (float)step;
        pz += sunDirZ * marchSize * (float)step;

        float lightSample = scene(px, py, pz, params, true);
        totalDensity += lightSample;
    }

    return BeersLaw(totalDensity, params->absorption);
}

// Main raymarching - matches original GLSL
static float raymarch(float roX, float roY, float roZ,
                      float rdX, float rdY, float rdZ,
                      CloudShaderParams *params) {
    float sunDirX = params->sun_x;
    float sunDirY = params->sun_y;
    float sunDirZ = params->sun_z;
    normalize3(&sunDirX, &sunDirY, &sunDirZ);

    float depth = 0.0f;
    float marchSize = 0.16f;  // MARCH_SIZE from original
    float px = roX + depth * rdX;
    float py = roY + depth * rdY;
    float pz = roZ + depth * rdZ;

    float totalTransmittance = 1.0f;
    float lightEnergy = 0.0f;

    float mu = dot3(rdX, rdY, rdZ, sunDirX, sunDirY, sunDirZ);
    float phase = HenyeyGreenstein(params->scattering, mu);

    for (int i = 0; i < params->march_steps; i++) {
        float density = scene(px, py, pz, params, false);

        // Only draw density if > 0
        if (density > 0.0f) {
            float lightTransmittance = lightmarch(px, py, pz, params, sunDirX, sunDirY, sunDirZ);
            float luminance = params->cloud_density * 0.025f + density * phase;

            totalTransmittance *= lightTransmittance;
            lightEnergy += totalTransmittance * luminance;
        }

        depth += marchSize;
        px = roX + depth * rdX;
        py = roY + depth * rdY;
        pz = roZ + depth * rdZ;
    }

    return clampf(lightEnergy, 0.0f, 1.0f);
}

// ============================================================================
// MAIN RENDER
// ============================================================================

static void render_clouds(
    PF_Pixel8 *in_data, PF_Pixel8 *out_data,
    int width, int height,
    int in_rowbytes, int out_rowbytes,
    CloudShaderParams *params
) {
    float sunDirX = params->sun_x;
    float sunDirY = params->sun_y;
    float sunDirZ = params->sun_z;
    normalize3(&sunDirX, &sunDirY, &sunDirZ);

    float aspectRatio = (float)width / (float)height;

    for (int y = 0; y < height; y++) {
        PF_Pixel8 *in_row = (PF_Pixel8*)((char*)in_data + y * in_rowbytes);
        PF_Pixel8 *out_row = (PF_Pixel8*)((char*)out_data + y * out_rowbytes);

        for (int x = 0; x < width; x++) {
            // UV coordinates centered at 0
            float u = ((float)x / (float)width - 0.5f) * aspectRatio;
            float v = (float)y / (float)height - 0.5f;

            // Ray Origin - camera
            float roX = 0.0f, roY = 0.0f, roZ = params->camera_dist;
            // Ray Direction
            float rdX = u, rdY = -v, rdZ = -1.0f;
            normalize3(&rdX, &rdY, &rdZ);

            // Base sky color - matches original GLSL
            float colorR = 0.7f;
            float colorG = 0.7f;
            float colorB = 0.9f;
            // Add vertical gradient
            colorR -= 0.8f * 0.9f * rdY;
            colorG -= 0.8f * 0.75f * rdY;
            colorB -= 0.8f * 0.9f * rdY;

            // Cloud
            float res = raymarch(roX, roY, roZ, rdX, rdY, rdZ, params);
            colorR += params->sun_r * res;
            colorG += params->sun_g * res;
            colorB += params->sun_b * res;

            // Clamp and blend
            colorR = clampf(colorR, 0.0f, 1.0f);
            colorG = clampf(colorG, 0.0f, 1.0f);
            colorB = clampf(colorB, 0.0f, 1.0f);

            PF_Pixel8 orig = in_row[x];
            out_row[x].red = (A_u_char)(mix(orig.red / 255.0f, colorR, 0.8f) * 255.0f);
            out_row[x].green = (A_u_char)(mix(orig.green / 255.0f, colorG, 0.8f) * 255.0f);
            out_row[x].blue = (A_u_char)(mix(orig.blue / 255.0f, colorB, 0.8f) * 255.0f);
            out_row[x].alpha = orig.alpha;
        }
    }
}

// ============================================================================
// PLUGIN ENTRY POINTS
// ============================================================================

static PF_Err About(PF_InData *in_data, PF_OutData *out_data,
                    PF_ParamDef *params[], PF_LayerDef *output) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "%s v%d.%d\r%s", NAME, MAJOR_VERSION, MINOR_VERSION, DESCRIPTION);
    return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_InData *in_data, PF_OutData *out_data,
                          PF_ParamDef *params[], PF_LayerDef *output) {
    out_data->my_version = PF_VERSION(MAJOR_VERSION, MINOR_VERSION, BUG_VERSION,
                                       STAGE_VERSION, BUILD_VERSION);
    // Render() consumes PF_Pixel8. Advertising deep-color awareness made AE
    // allocate and hand us 16-bpc buffers that this implementation cannot use.
    out_data->out_flags = PF_OutFlag_PIX_INDEPENDENT;
    // Topics use START_COLLAPSED selectively, keeping the common controls
    // visible while advanced sections stay compact in AE's Effect Controls.
    out_data->out_flags2 = PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG;
    // Shared D3D resources remain serialized by g_gpuMutex in Render().
    return PF_Err_NONE;
}

static PF_Err GlobalSetdown(PF_InData *in_data, PF_OutData *out_data,
                            PF_ParamDef *params[], PF_LayerDef *output) {
    // Clean up GPU resources
    std::lock_guard<std::mutex> lock(g_gpuMutex);
    ReleaseGPUResources();
    if (g_noiseSampler) { g_noiseSampler->Release(); g_noiseSampler = nullptr; }
    if (g_noiseUAV) { g_noiseUAV->Release(); g_noiseUAV = nullptr; }
    if (g_noiseSRV) { g_noiseSRV->Release(); g_noiseSRV = nullptr; }
    if (g_noiseTex) { g_noiseTex->Release(); g_noiseTex = nullptr; }
    if (g_sharpenShader) { g_sharpenShader->Release(); g_sharpenShader = nullptr; }
    if (g_noiseShader) { g_noiseShader->Release(); g_noiseShader = nullptr; }
    if (g_computeShader) { g_computeShader->Release(); g_computeShader = nullptr; }
    if (g_context) { g_context->Release(); g_context = nullptr; }
    if (g_device) { g_device->Release(); g_device = nullptr; }
    g_gpuInitialized = false;
    g_gpuFailed = false;
    return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData *in_data, PF_OutData *out_data,
                          PF_ParamDef *params[], PF_LayerDef *output) {
    PF_ParamDef def;

    PF_ADD_TOPICX("Volume & Render Quality", PF_ParamFlag_START_COLLAPSED,
                  CLOUD_VOLUME_GROUP_DISK_ID);

    // A larger default produces a useful hero-cloud framing in a 16:9 comp.
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Layer Thickness", 0.1, 5.0, 0.1, 5.0, 2.6,
                         PF_Precision_HUNDREDTHS, 0, 0, CLOUD_RADIUS_DISK_ID);

    // Physical density multiplier; it drives extinction and opacity.
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Coverage / Density", 0.1, 10.0, 0.1, 10.0, 3.0,
                         PF_Precision_HUNDREDTHS, 0, 0, CLOUD_DENSITY_DISK_ID);

    // Default: 0.9 (ABSORPTION_COEFFICIENT in original)
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Absorption", 0.0, 2.0, 0.0, 2.0, 1.0,
                         PF_Precision_HUNDREDTHS, 0, 0, ABSORPTION_DISK_ID);

    // Henyey-Greenstein anisotropy. Values near 0.4 give natural silver lining.
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Scattering", -0.85, 0.85, -0.85, 0.85, 0.25,
                         PF_Precision_HUNDREDTHS, 0, 0, SCATTERING_DISK_ID);

    // The shader intersects the volume first, so every step now contributes.
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("March Steps", 8, 120, 8, 120, 80,
                         PF_Precision_INTEGER, 0, 0, MARCH_STEPS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Light Steps", 2, 16, 2, 16, 10,
                         PF_Precision_INTEGER, 0, 0, LIGHT_STEPS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CLOUD_VOLUME_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Manual Sun", PF_ParamFlag_START_COLLAPSED,
                  MANUAL_SUN_GROUP_DISK_ID);

    // Sun position: (2.0, 1.0, 2.0) in original
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sun X", -10.0, 10.0, -10.0, 10.0, 3.0,
                         PF_Precision_HUNDREDTHS, 0, 0, SUN_X_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sun Y", -10.0, 10.0, -10.0, 10.0, 1.5,
                         PF_Precision_HUNDREDTHS, 0, 0, SUN_Y_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sun Z", -10.0, 10.0, -10.0, 10.0, 1.5,
                         PF_Precision_HUNDREDTHS, 0, 0, SUN_Z_DISK_ID);

    // Neutral-warm daylight; the previous saturated orange default tinted the
    // entire cloud rather than just the terminator and silver lining.
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Sun Color", 255, 242, 219, SUN_COLOR_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(MANUAL_SUN_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Camera", PF_ParamFlag_START_COLLAPSED,
                  CAMERA_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Focal Length", 1.0, 10.0, 1.0, 10.0, 3.0,
                         PF_Precision_HUNDREDTHS, 0, 0, CAMERA_DIST_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Pitch", -89.0, 90.0, -89.0, 90.0, 90.0,
                         PF_Precision_HUNDREDTHS, 0, 0, CAMERA_PITCH_DISK_ID);

    // Angle controls are cyclic and can be spun/keyframed beyond one turn.
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Yaw", 0.0, CAMERA_YAW_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CAMERA_GROUP_END_DISK_ID);

    // This is the primary working section, so it starts open.
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC("Cloud Shape & Motion", CLOUD_SHAPE_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Wind Speed", 0.0, 500, 0, 500, 100,
                         PF_Precision_INTEGER, 0, 0, WIND_SPEED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Noise Scale", 0.1, 5.0, 0.1, 5.0, 2.4,
                         PF_Precision_HUNDREDTHS, 0, 0, NOISE_SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Detail", 1, 8, 1, 8, 6,
                         PF_Precision_INTEGER, 0, 0, DETAIL_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Cloud Coverage", 0.0, 100.0, 0.0, 100.0, 57.0,
                         PF_Precision_HUNDREDTHS, 0, 0, CLOUD_COVERAGE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Cloud Type", 0.0, 100.0, 0.0, 100.0, 88.0,
                         PF_Precision_HUNDREDTHS, 0, 0, CLOUD_TYPE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Edge Erosion", 0.0, 200.0, 0.0, 200.0, 72.0,
                         PF_Precision_HUNDREDTHS, 0, 0, EROSION_STRENGTH_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Billow Size", 0.25, 3.0, 0.25, 3.0, 1.15,
                         PF_Precision_HUNDREDTHS, 0, 0, BILLOW_SIZE_DISK_ID);

    // Wind direction is a true AE angle dial with unlimited revolutions.
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Wind Direction", -35.0, WIND_DIRECTION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Altitude Wind Shear", 0.0, 200.0, 0.0, 200.0, 70.0,
                         PF_Precision_HUNDREDTHS, 0, 0, WIND_SHEAR_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Turbulent Evolution", 0.0, 200.0, 0.0, 200.0, 80.0,
                         PF_Precision_HUNDREDTHS, 0, 0, WIND_TURBULENCE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CLOUD_SHAPE_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Cloud Types & Layers", PF_ParamFlag_START_COLLAPSED,
                  CLOUD_LAYERS_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(
        "Cloud Regime", 11, 1,
        "Mixed Weather|Cirrus|Cirrostratus|Cirrocumulus|Altocumulus|Altostratus|Nimbostratus|Stratocumulus|Stratus|Cumulus|Cumulonimbus",
        CLOUD_REGIME_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Mid-Level Clouds", 0.0, 100.0, 0.0, 100.0, 32.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MID_LEVEL_AMOUNT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("High-Level Clouds", 0.0, 100.0, 0.0, 100.0, 20.0,
                         PF_Precision_HUNDREDTHS, 0, 0, HIGH_LEVEL_AMOUNT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Storm Development", 0.0, 100.0, 0.0, 100.0, 5.0,
                         PF_Precision_HUNDREDTHS, 0, 0, STORM_DEVELOPMENT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(CLOUD_LAYERS_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Hero Cloud Tower", PF_ParamFlag_START_COLLAPSED,
                  HERO_TOWER_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tower Development", 0.0, 200.0, 0.0, 200.0, 0.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TOWER_DEVELOPMENT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tower Width", 25.0, 300.0, 25.0, 300.0, 120.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TOWER_SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Cauliflower Detail", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, CAULIFLOWER_DETAIL_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tower Position X", -100.0, 100.0, -100.0, 100.0, 0.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TOWER_POSITION_X_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tower Distance", 1.0, 100.0, 1.0, 100.0, 18.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TOWER_DISTANCE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tower Isolation", 0.0, 100.0, 0.0, 100.0, 70.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TOWER_ISOLATION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(HERO_TOWER_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Rain & Virga", PF_ParamFlag_START_COLLAPSED,
                  RAIN_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Rain", "", FALSE, 0, RAIN_ENABLED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rain Amount", 0.0, 200.0, 0.0, 200.0, 75.0,
                         PF_Precision_HUNDREDTHS, 0, 0, RAIN_AMOUNT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rain Prevalence", 0.0, 100.0, 0.0, 100.0, 42.0,
                         PF_Precision_HUNDREDTHS, 0, 0, RAIN_PREVALENCE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rain Shaft Detail", 0.0, 100.0, 0.0, 100.0, 62.0,
                         PF_Precision_HUNDREDTHS, 0, 0, RAIN_SHAFT_DETAIL_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rain Mist", 0.0, 100.0, 0.0, 100.0, 45.0,
                         PF_Precision_HUNDREDTHS, 0, 0, RAIN_MIST_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Virga / Evaporation", 0.0, 100.0, 0.0, 100.0, 20.0,
                         PF_Precision_HUNDREDTHS, 0, 0, RAIN_EVAPORATION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Rain Fall Speed", 0.0, 300.0, 0.0, 300.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, RAIN_FALL_SPEED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(RAIN_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Day Cycle & Sky", PF_ParamFlag_START_COLLAPSED,
                  DAY_SKY_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Time of Day", "", TRUE, 0, DAY_CYCLE_ENABLED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Time of Day", 0.0, 24.0, 0.0, 24.0, 12.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TIME_OF_DAY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Day Cycle Speed", -24.0, 24.0, -24.0, 24.0, 0.0,
                         PF_Precision_HUNDREDTHS, 0, 0, DAY_CYCLE_SPEED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Sun Path Rotation", 0.0, SUN_PATH_ROTATION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sun Peak Elevation", 5.0, 90.0, 5.0, 90.0, 65.0,
                         PF_Precision_HUNDREDTHS, 0, 0, SUN_PEAK_ELEVATION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sky Vibrancy", 0.0, 200.0, 0.0, 200.0, 130.0,
                         PF_Precision_HUNDREDTHS, 0, 0, SKY_VIBRANCY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sky Exposure", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, SKY_EXPOSURE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Twilight Vibrancy", 0.0, 200.0, 0.0, 200.0, 130.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TWILIGHT_INTENSITY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Twilight Range", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, TWILIGHT_RANGE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Horizon Haze", 0.0, 200.0, 0.0, 200.0, 80.0,
                         PF_Precision_HUNDREDTHS, 0, 0, HORIZON_HAZE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sun Disc Size", 10.0, 500.0, 10.0, 500.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, SUN_DISK_SIZE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Sun Brightness", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, SUN_BRIGHTNESS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Night Brightness", 0.0, 100.0, 0.0, 100.0, 12.0,
                         PF_Precision_HUNDREDTHS, 0, 0, NIGHT_BRIGHTNESS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(DAY_SKY_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Stars", PF_ParamFlag_START_COLLAPSED,
                  STARS_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Stars", "", TRUE, 0, STARS_ENABLED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Star Amount", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, STAR_AMOUNT_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Star Brightness", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, STAR_BRIGHTNESS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Star Twinkle", 0.0, 100.0, 0.0, 100.0, 30.0,
                         PF_Precision_HUNDREDTHS, 0, 0, STAR_TWINKLE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(STARS_GROUP_END_DISK_ID);

    PF_ADD_TOPICX("Moon", PF_ParamFlag_START_COLLAPSED,
                  MOON_GROUP_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Enable Moon", "", TRUE, 0, MOON_ENABLED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Moon Azimuth", -38.0, MOON_AZIMUTH_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Elevation", -89.0, 89.0, -89.0, 89.0, 42.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_ELEVATION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Drift Speed", -30.0, 30.0, -30.0, 30.0, 0.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_DRIFT_SPEED_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Size", 10.0, 3000.0, 10.0, 3000.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_SIZE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Phase", 0.0, 100.0, 0.0, 100.0, 50.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_PHASE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Brightness", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_BRIGHTNESS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Glow", 0.0, 200.0, 0.0, 200.0, 55.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_GLOW_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Surface Detail", 0.0, 200.0, 0.0, 200.0, 100.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_SURFACE_DETAIL_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Moon Earthshine", 0.0, 100.0, 0.0, 100.0, 6.0,
                         PF_Precision_HUNDREDTHS, 0, 0, MOON_EARTHSHINE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(MOON_GROUP_END_DISK_ID);

    out_data->num_params = NUM_PARAMS;
    return PF_Err_NONE;
}

static PF_Err Render(PF_InData *in_data, PF_OutData *out_data,
                     PF_ParamDef *params[], PF_LayerDef *output) {
    PF_LayerDef *input = &params[INPUT_LAYER]->u.ld;

    // Get parameters
    float cloud_radius = (float)params[CLOUD_RADIUS_PARAM]->u.fs_d.value;
    float cloud_density = (float)params[CLOUD_DENSITY_PARAM]->u.fs_d.value;
    float absorption = (float)params[ABSORPTION_PARAM]->u.fs_d.value;
    float scattering = (float)params[SCATTERING_PARAM]->u.fs_d.value;
    int march_steps = (int)params[MARCH_STEPS_PARAM]->u.fs_d.value;
    int light_steps = (int)params[LIGHT_STEPS_PARAM]->u.fs_d.value;
    float sun_x = (float)params[SUN_X_PARAM]->u.fs_d.value;
    float sun_y = (float)params[SUN_Y_PARAM]->u.fs_d.value;
    float sun_z = (float)params[SUN_Z_PARAM]->u.fs_d.value;

    PF_Pixel sun_color = params[SUN_COLOR_PARAM]->u.cd.value;
    float sun_r = sun_color.red / 255.0f;
    float sun_g = sun_color.green / 255.0f;
    float sun_b = sun_color.blue / 255.0f;

    float wind_speed = (float)params[WIND_SPEED_PARAM]->u.fs_d.value / 100.0f;
    float noise_scale = (float)params[NOISE_SCALE_PARAM]->u.fs_d.value;
    int detail = (int)params[DETAIL_PARAM]->u.fs_d.value;
    float camera_dist = (float)params[CAMERA_DIST_PARAM]->u.fs_d.value;
    float camera_pitch = (float)params[CAMERA_PITCH_PARAM]->u.fs_d.value;
    float camera_yaw = (float)FIX_2_FLOAT(
        params[CAMERA_YAW_PARAM]->u.ad.value);
    float cloud_coverage = (float)params[CLOUD_COVERAGE_PARAM]->u.fs_d.value / 100.0f;
    float cloud_type = (float)params[CLOUD_TYPE_PARAM]->u.fs_d.value / 100.0f;
    float erosion_strength = (float)params[EROSION_STRENGTH_PARAM]->u.fs_d.value / 100.0f;
    float billow_size = (float)params[BILLOW_SIZE_PARAM]->u.fs_d.value;
    float wind_direction = (float)FIX_2_FLOAT(
        params[WIND_DIRECTION_PARAM]->u.ad.value);
    float wind_shear = (float)params[WIND_SHEAR_PARAM]->u.fs_d.value / 100.0f;
    float wind_turbulence = (float)params[WIND_TURBULENCE_PARAM]->u.fs_d.value / 100.0f;
    int cloud_regime = (int)params[CLOUD_REGIME_PARAM]->u.pd.value;
    float mid_level_amount =
        (float)params[MID_LEVEL_AMOUNT_PARAM]->u.fs_d.value / 100.0f;
    float high_level_amount =
        (float)params[HIGH_LEVEL_AMOUNT_PARAM]->u.fs_d.value / 100.0f;
    float storm_development =
        (float)params[STORM_DEVELOPMENT_PARAM]->u.fs_d.value / 100.0f;
    float tower_development =
        (float)params[TOWER_DEVELOPMENT_PARAM]->u.fs_d.value / 100.0f;
    float tower_scale =
        (float)params[TOWER_SCALE_PARAM]->u.fs_d.value / 100.0f;
    float cauliflower_detail =
        (float)params[CAULIFLOWER_DETAIL_PARAM]->u.fs_d.value / 100.0f;
    float tower_position_x =
        (float)params[TOWER_POSITION_X_PARAM]->u.fs_d.value;
    float tower_distance =
        (float)params[TOWER_DISTANCE_PARAM]->u.fs_d.value;
    float tower_isolation =
        (float)params[TOWER_ISOLATION_PARAM]->u.fs_d.value / 100.0f;
    int rain_enabled = params[RAIN_ENABLED_PARAM]->u.bd.value ? 1 : 0;
    float rain_amount =
        (float)params[RAIN_AMOUNT_PARAM]->u.fs_d.value / 100.0f;
    float rain_prevalence =
        (float)params[RAIN_PREVALENCE_PARAM]->u.fs_d.value / 100.0f;
    float rain_shaft_detail =
        (float)params[RAIN_SHAFT_DETAIL_PARAM]->u.fs_d.value / 100.0f;
    float rain_mist =
        (float)params[RAIN_MIST_PARAM]->u.fs_d.value / 100.0f;
    float rain_evaporation =
        (float)params[RAIN_EVAPORATION_PARAM]->u.fs_d.value / 100.0f;
    float rain_fall_speed =
        (float)params[RAIN_FALL_SPEED_PARAM]->u.fs_d.value / 100.0f;
    int day_cycle_enabled =
        params[DAY_CYCLE_ENABLED_PARAM]->u.bd.value ? 1 : 0;
    float time_of_day =
        (float)params[TIME_OF_DAY_PARAM]->u.fs_d.value;
    float day_cycle_speed =
        (float)params[DAY_CYCLE_SPEED_PARAM]->u.fs_d.value;
    float sun_path_rotation = (float)FIX_2_FLOAT(
        params[SUN_PATH_ROTATION_PARAM]->u.ad.value);
    float sun_peak_elevation =
        (float)params[SUN_PEAK_ELEVATION_PARAM]->u.fs_d.value;
    float sky_vibrancy =
        (float)params[SKY_VIBRANCY_PARAM]->u.fs_d.value / 100.0f;
    float sky_exposure =
        (float)params[SKY_EXPOSURE_PARAM]->u.fs_d.value / 100.0f;
    float twilight_intensity =
        (float)params[TWILIGHT_INTENSITY_PARAM]->u.fs_d.value / 100.0f;
    float twilight_range =
        (float)params[TWILIGHT_RANGE_PARAM]->u.fs_d.value / 100.0f;
    float horizon_haze =
        (float)params[HORIZON_HAZE_PARAM]->u.fs_d.value / 100.0f;
    float sun_disk_size =
        (float)params[SUN_DISK_SIZE_PARAM]->u.fs_d.value / 100.0f;
    float sun_brightness =
        (float)params[SUN_BRIGHTNESS_PARAM]->u.fs_d.value / 100.0f;
    float night_brightness =
        (float)params[NIGHT_BRIGHTNESS_PARAM]->u.fs_d.value / 100.0f;
    int stars_enabled = params[STARS_ENABLED_PARAM]->u.bd.value ? 1 : 0;
    float star_amount =
        (float)params[STAR_AMOUNT_PARAM]->u.fs_d.value / 100.0f;
    float star_brightness =
        (float)params[STAR_BRIGHTNESS_PARAM]->u.fs_d.value / 100.0f;
    float star_twinkle =
        (float)params[STAR_TWINKLE_PARAM]->u.fs_d.value / 100.0f;
    int moon_enabled = params[MOON_ENABLED_PARAM]->u.bd.value ? 1 : 0;
    float moon_azimuth = (float)FIX_2_FLOAT(
        params[MOON_AZIMUTH_PARAM]->u.ad.value);
    float moon_elevation =
        (float)params[MOON_ELEVATION_PARAM]->u.fs_d.value;
    float moon_drift_speed =
        (float)params[MOON_DRIFT_SPEED_PARAM]->u.fs_d.value;
    float moon_size =
        (float)params[MOON_SIZE_PARAM]->u.fs_d.value / 100.0f;
    float moon_phase =
        (float)params[MOON_PHASE_PARAM]->u.fs_d.value / 100.0f;
    float moon_brightness =
        (float)params[MOON_BRIGHTNESS_PARAM]->u.fs_d.value / 100.0f;
    float moon_glow =
        (float)params[MOON_GLOW_PARAM]->u.fs_d.value / 100.0f;
    float moon_surface_detail =
        (float)params[MOON_SURFACE_DETAIL_PARAM]->u.fs_d.value / 100.0f;
    float moon_earthshine =
        (float)params[MOON_EARTHSHINE_PARAM]->u.fs_d.value / 100.0f;
    // Resolve the selected meteorological regime once on the CPU. The shader
    // can then sample hundreds of density points without repeating this switch.
    float resolved_low_amount = 1.0f;
    float resolved_mid_amount = mid_level_amount;
    float resolved_high_amount = high_level_amount;
    float resolved_storm = storm_development;
    float resolved_low_type = cloud_type;
    float resolved_mid_cellular = 0.68f;
    float resolved_low_coverage = 0.0f;
    float resolved_mid_coverage = 0.0f;
    float resolved_erosion = erosion_strength;
    float resolved_billow_size = billow_size;
    float resolved_rain_factor = 0.68f;
    float resolved_tower_development = tower_development;
    switch (cloud_regime) {
        case 2: // Cirrus
            resolved_low_amount = 0.02f; resolved_mid_amount = 0.02f;
            resolved_high_amount = (std::max)(high_level_amount, 0.88f);
            resolved_storm = 0.0f; resolved_rain_factor = 0.0f;
            resolved_tower_development = 0.0f; break;
        case 3: // Cirrostratus
            resolved_low_amount = 0.03f; resolved_mid_amount = 0.08f;
            resolved_high_amount = (std::max)(high_level_amount, 0.82f);
            resolved_storm = 0.0f; resolved_rain_factor = 0.02f;
            resolved_tower_development = 0.0f; break;
        case 4: // Cirrocumulus
            resolved_low_amount = 0.03f; resolved_mid_amount = 0.10f;
            resolved_high_amount = (std::max)(high_level_amount, 0.84f);
            resolved_storm = 0.0f; resolved_rain_factor = 0.08f;
            resolved_tower_development = 0.0f; break;
        case 5: // Altocumulus
            resolved_low_amount = 0.08f;
            resolved_mid_amount = (std::max)(mid_level_amount, 0.86f);
            resolved_high_amount = 0.08f; resolved_storm = 0.02f;
            resolved_mid_cellular = 1.0f; resolved_rain_factor = 0.32f;
            resolved_tower_development = 0.0f; break;
        case 6: // Altostratus
            resolved_low_amount = 0.12f;
            resolved_mid_amount = (std::max)(mid_level_amount, 0.90f);
            resolved_high_amount = 0.12f; resolved_storm = 0.02f;
            resolved_mid_cellular = 0.0f; resolved_mid_coverage = 0.17f;
            resolved_rain_factor = 0.72f;
            resolved_tower_development = 0.0f; break;
        case 7: // Nimbostratus
            resolved_low_amount = 0.88f;
            resolved_mid_amount = (std::max)(mid_level_amount, 0.96f);
            resolved_high_amount = (std::max)(high_level_amount, 0.22f);
            resolved_storm = (std::max)(storm_development, 0.14f);
            resolved_low_type = 0.04f; resolved_mid_cellular = 0.0f;
            resolved_low_coverage = 0.22f; resolved_mid_coverage = 0.28f;
            resolved_erosion = (std::min)(erosion_strength, 0.46f);
            resolved_billow_size = (std::max)(billow_size, 1.35f);
            resolved_rain_factor = 1.15f;
            resolved_tower_development = (std::min)(tower_development, 0.10f); break;
        case 8: // Stratocumulus
            resolved_mid_amount = (std::max)(mid_level_amount, 0.14f);
            resolved_high_amount = 0.04f; resolved_storm = 0.04f;
            resolved_low_type = 0.32f; resolved_low_coverage = 0.12f;
            resolved_rain_factor = 0.42f;
            resolved_tower_development = 0.0f; break;
        case 9: // Stratus
            resolved_mid_amount = 0.06f; resolved_high_amount = 0.03f;
            resolved_storm = 0.0f; resolved_low_type = 0.04f;
            resolved_low_coverage = 0.22f;
            resolved_erosion = (std::min)(erosion_strength, 0.38f);
            resolved_billow_size = (std::max)(billow_size, 1.45f);
            resolved_rain_factor = 0.32f;
            resolved_tower_development = 0.0f; break;
        case 10: // Cumulus
            resolved_mid_amount = 0.08f; resolved_high_amount = 0.04f;
            resolved_storm = (std::max)(storm_development, 0.12f);
            resolved_low_type = 0.82f; resolved_low_coverage = -0.05f;
            resolved_rain_factor = 0.62f;
            resolved_tower_development = (std::max)(tower_development, 0.28f); break;
        case 11: // Cumulonimbus
            resolved_mid_amount = (std::max)(mid_level_amount, 0.12f);
            resolved_high_amount = (std::max)(high_level_amount, 0.34f);
            resolved_storm = (std::max)(storm_development, 0.95f);
            resolved_low_type = 1.0f; resolved_low_coverage = -0.05f;
            resolved_erosion = (std::min)(erosion_strength, 0.48f);
            resolved_billow_size = (std::max)(billow_size, 1.45f);
            resolved_rain_factor = 1.25f;
            resolved_tower_development = (std::max)(tower_development, 1.15f); break;
        default: break;
    }
    if (cloud_regime == 1) {
        resolved_rain_factor += resolved_storm * 0.35f;
    }
    // Time remains physical seconds. HLSL applies wind speed exactly once.
    float time = (float)in_data->current_time / (float)in_data->time_scale;

    // A great-circle solar path gives stable sunrise/noon/sunset positions.
    // Peak elevation acts like a simple latitude control; path rotation turns
    // east/west around the horizon. A zero cycle speed leaves Time of Day fully
    // keyframeable, while non-zero values animate hours per comp second.
    if (day_cycle_enabled != 0) {
        float resolved_hours = fmodf(time_of_day + time * day_cycle_speed, 24.0f);
        if (resolved_hours < 0.0f) resolved_hours += 24.0f;
        float hour_angle = (resolved_hours - 12.0f) * (PI / 12.0f);
        float latitude = (90.0f - clampf(sun_peak_elevation, 5.0f, 90.0f))
                       * (PI / 180.0f);
        float east = -sinf(hour_angle);
        float north = -cosf(hour_angle) * sinf(latitude);
        float up = cosf(hour_angle) * cosf(latitude);
        float rotation = sun_path_rotation * (PI / 180.0f);
        sun_x = east * cosf(rotation) - north * sinf(rotation);
        sun_z = east * sinf(rotation) + north * cosf(rotation);
        sun_y = up;
    }

    float sun_length = sqrtf(sun_x * sun_x + sun_y * sun_y + sun_z * sun_z);
    float normalized_sun_y = sun_length > 0.0001f ? sun_y / sun_length : 0.0f;
    float daylight = smoothstepHost(-0.12f, 0.045f, normalized_sun_y);
    float high_sun = smoothstepHost(0.02f, 0.45f, normalized_sun_y);
    sun_r = mix(1.00f, sun_r, high_sun) * daylight;
    sun_g = mix(0.55f, sun_g, high_sun) * daylight;
    sun_b = mix(0.25f, sun_b, high_sun) * daylight;

    // Try GPU rendering first
    GPUCloudParams gpu_params;
    gpu_params.cloud_radius = cloud_radius;
    gpu_params.cloud_density = cloud_density;
    gpu_params.absorption = absorption;
    gpu_params.scattering = scattering;
    gpu_params.march_steps = march_steps;
    gpu_params.light_steps = light_steps;
    gpu_params.sun_x = sun_x;
    gpu_params.sun_y = sun_y;
    gpu_params.sun_z = sun_z;
    gpu_params.sun_r = sun_r;
    gpu_params.sun_g = sun_g;
    gpu_params.sun_b = sun_b;
    gpu_params.wind_speed = wind_speed;
    gpu_params.noise_scale = noise_scale;
    gpu_params.detail = detail;
    gpu_params.camera_dist = camera_dist;
    gpu_params.time = time;
    gpu_params.width = output->width;
    gpu_params.height = output->height;
    gpu_params.camera_pitch = camera_pitch;
    gpu_params.camera_yaw = camera_yaw;
    float wind_direction_radians = wind_direction * (PI / 180.0f);
    gpu_params.wind_direction_x = cosf(wind_direction_radians);
    gpu_params.wind_direction_z = sinf(wind_direction_radians);
    gpu_params.wind_shear = wind_shear;
    gpu_params.wind_turbulence = wind_turbulence;
    gpu_params.cloud_coverage = cloud_coverage;
    gpu_params.cloud_type = resolved_low_type;
    gpu_params.erosion_strength = resolved_erosion;
    gpu_params.billow_size = resolved_billow_size;
    gpu_params.mid_level_amount = resolved_mid_amount;
    gpu_params.high_level_amount = resolved_high_amount;
    gpu_params.storm_development = resolved_storm;
    gpu_params.cloud_regime = cloud_regime;
    gpu_params.low_level_amount = resolved_low_amount;
    gpu_params.mid_cellular_style = resolved_mid_cellular;
    gpu_params.low_coverage_offset = resolved_low_coverage;
    gpu_params.mid_coverage_offset = resolved_mid_coverage;
    gpu_params.tower_development = resolved_tower_development;
    gpu_params.tower_scale = tower_scale;
    gpu_params.cauliflower_detail = cauliflower_detail;
    gpu_params.tower_position_x = tower_position_x;
    gpu_params.tower_distance = tower_distance;
    gpu_params.tower_isolation = tower_isolation;
    gpu_params.tower_padding_0 = 0.0f;
    gpu_params.rain_enabled = rain_enabled;
    gpu_params.rain_amount = rain_amount;
    gpu_params.rain_prevalence = rain_prevalence;
    gpu_params.rain_shaft_detail = rain_shaft_detail;
    gpu_params.rain_mist = rain_mist;
    gpu_params.rain_evaporation = rain_evaporation;
    gpu_params.rain_fall_speed = rain_fall_speed;
    gpu_params.rain_regime_factor = resolved_rain_factor;
    gpu_params.sky_vibrancy = sky_vibrancy;
    gpu_params.sky_exposure = sky_exposure;
    gpu_params.twilight_intensity = twilight_intensity;
    gpu_params.twilight_range = twilight_range;
    gpu_params.horizon_haze = horizon_haze;
    gpu_params.sun_disk_size = sun_disk_size;
    gpu_params.sun_brightness = sun_brightness;
    gpu_params.night_brightness = night_brightness;
    gpu_params.stars_enabled = stars_enabled;
    gpu_params.star_amount = star_amount;
    gpu_params.star_brightness = star_brightness;
    gpu_params.star_twinkle = star_twinkle;
    gpu_params.moon_enabled = moon_enabled;
    gpu_params.moon_azimuth = moon_azimuth;
    gpu_params.moon_elevation = moon_elevation;
    gpu_params.moon_size = moon_size;
    gpu_params.moon_phase = moon_phase;
    gpu_params.moon_brightness = moon_brightness;
    gpu_params.moon_glow = moon_glow;
    gpu_params.moon_drift_speed = moon_drift_speed;
    gpu_params.moon_surface_detail = moon_surface_detail;
    gpu_params.moon_earthshine = moon_earthshine;
    gpu_params.sky_padding_0 = 0.0f;
    gpu_params.sky_padding_1 = 0.0f;

    // Try GPU rendering, fall back to CPU if it fails
    bool gpuSuccess = RenderOnGPU(
        (PF_Pixel8*)input->data,
        (PF_Pixel8*)output->data,
        output->width,
        output->height,
        input->rowbytes,
        output->rowbytes,
        &gpu_params
    );

    if (!gpuSuccess) {
        // CPU fallback
        CloudShaderParams cpu_params;
        cpu_params.cloud_radius = cloud_radius;
        // The legacy CPU emergency path predates the physical density model.
        // Keep its historical luminance scale; normal operation uses hardware
        // DirectCompute or WARP and therefore the exact HLSL previewed offline.
        cpu_params.cloud_density = cloud_density * 0.01f;
        cpu_params.absorption = absorption;
        cpu_params.scattering = scattering;
        cpu_params.march_steps = march_steps;
        cpu_params.light_steps = light_steps;
        cpu_params.sun_x = sun_x;
        cpu_params.sun_y = sun_y;
        cpu_params.sun_z = sun_z;
        cpu_params.sun_r = sun_r;
        cpu_params.sun_g = sun_g;
        cpu_params.sun_b = sun_b;
        cpu_params.wind_speed = wind_speed;
        cpu_params.noise_scale = noise_scale;
        cpu_params.detail = detail;
        cpu_params.camera_dist = camera_dist;
        cpu_params.camera_pitch = camera_pitch;
        cpu_params.camera_yaw = camera_yaw;
        cpu_params.wind_direction = wind_direction;
        cpu_params.wind_shear = wind_shear;
        cpu_params.wind_turbulence = wind_turbulence;
        cpu_params.cloud_coverage = cloud_coverage;
        cpu_params.cloud_type = resolved_low_type;
        cpu_params.erosion_strength = resolved_erosion;
        cpu_params.billow_size = resolved_billow_size;
        cpu_params.mid_level_amount = resolved_mid_amount;
        cpu_params.high_level_amount = resolved_high_amount;
        cpu_params.storm_development = resolved_storm;
        cpu_params.cloud_regime = cloud_regime;
        cpu_params.rain_enabled = rain_enabled;
        cpu_params.rain_amount = rain_amount;
        cpu_params.rain_prevalence = rain_prevalence;
        cpu_params.rain_shaft_detail = rain_shaft_detail;
        cpu_params.rain_mist = rain_mist;
        cpu_params.rain_evaporation = rain_evaporation;
        cpu_params.rain_fall_speed = rain_fall_speed;
        cpu_params.time = time;

        render_clouds(
            (PF_Pixel8*)input->data,
            (PF_Pixel8*)output->data,
            output->width,
            output->height,
            input->rowbytes,
            output->rowbytes,
            &cpu_params
        );
    }

    return PF_Err_NONE;
}

DllExport PF_Err EntryPointFunc(
    PF_Cmd cmd, PF_InData *in_data, PF_OutData *out_data,
    PF_ParamDef *params[], PF_LayerDef *output, void *extra
) {
    PF_Err err = PF_Err_NONE;

    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:
                err = About(in_data, out_data, params, output);
                break;
            case PF_Cmd_GLOBAL_SETUP:
                err = GlobalSetup(in_data, out_data, params, output);
                break;
            case PF_Cmd_GLOBAL_SETDOWN:
                err = GlobalSetdown(in_data, out_data, params, output);
                break;
            case PF_Cmd_PARAMS_SETUP:
                err = ParamsSetup(in_data, out_data, params, output);
                break;
            case PF_Cmd_RENDER:
                err = Render(in_data, out_data, params, output);
                break;
        }
    } catch (PF_Err &thrown_err) {
        err = thrown_err;
    }

    return err;
}

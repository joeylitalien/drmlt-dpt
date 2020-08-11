#include "scene.h"
#include "distribution.h"
#include "shape.h"
#include "light.h"
#include "camera.h"
#include "bounds.h"
#include "xmmintrin.h"
#include "pmmintrin.h"

Scene::Scene(const std::shared_ptr<DptOptions> &options,
             const std::shared_ptr<const Camera> &camera,
             const std::vector<std::shared_ptr<const Shape>> &objects,
             const std::vector<std::shared_ptr<const Light>> &lights,
             const std::shared_ptr<const EnvLight> &envLight,
             const std::string &outputName)
    : options(options),
      camera(camera),
      objects(objects),
      lights(lights),
      envLight(envLight),
      outputName(outputName) {

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);


    // Build light cdf
    std::vector<Float> weights(lights.size());
    lightWeightSum = Float(0.0);
    for (size_t i = 0; i < weights.size(); i++) {
        weights[i] = lights[i]->samplingWeight;
        lightWeightSum += weights[i];
    }
    lightDist = std::unique_ptr<PiecewiseConstant1D>(new PiecewiseConstant1D(&weights[0], weights.size()));

    /*
    * threads=[int]: Specifies a number of build threads to use. A value of 0 enables all detected hardware threads. By default all hardware threads are used.
    * set_affinity=[0/1]: When enabled, build threads are affinitized to hardware threads. This option is disabled by default on standard CPUs, and enabled by default on Xeon Phi Processors.
    * start_threads=[0/1]: When enabled, the build threads are started upfront. This can be useful for benchmarking to exclude thread creation time. This option is disabled by default.
    * verbose=[0,1,2,3]: Sets the verbosity of the output. When set to 0, no output is printed by Embree, when set to a higher level more output is printed. By default Embree does not print anything on the console.
    */
    rtcDevice = rtcNewDevice("start_threads=1,verbose=3, threads=0"); // set_affinity=1
    rtcScene = rtcDeviceNewScene(
        rtcDevice,
        RTC_SCENE_STATIC | RTC_SCENE_INCOHERENT | RTC_SCENE_HIGH_QUALITY | RTC_SCENE_ROBUST,
        RTC_INTERSECT1);
    BBox bbox;
    for (auto &obj : objects) {
        obj->RtcRegister(rtcScene);
        bbox = Merge(bbox, obj->GetBBox());
    }
    bSphere = BSphere(bbox);
    rtcCommit(rtcScene);
}

Scene::~Scene() {
    rtcDeleteScene(rtcScene);
    rtcDeleteDevice(rtcDevice);
}

static inline RTCRay ToRTCRay(const Float time, const RaySegment &raySeg) {
    const Ray &ray = raySeg.ray;
    RTCRay rtcRay;
    for (int i = 0; i < 3; i++) {
        assert(std::isfinite(ray.org[i]));
        assert(std::isfinite(ray.dir[i]));
        rtcRay.org[i] = float(ray.org[i]);
        rtcRay.dir[i] = float(ray.dir[i]);
    }
    rtcRay.tnear = float(raySeg.minT);
    rtcRay.tfar = float(raySeg.maxT);
    rtcRay.geomID = RTC_INVALID_GEOMETRY_ID;
    rtcRay.primID = RTC_INVALID_GEOMETRY_ID;
    rtcRay.instID = RTC_INVALID_GEOMETRY_ID;
    rtcRay.mask = 0xFFFFFFFF;
    rtcRay.time = float(time);
    return rtcRay;
}

bool Intersect(const Scene *scene,
               const Float time,
               const RaySegment &raySeg,
               ShapeInst &shapeInst) {
    RTCRay rtcRay = ToRTCRay(time, raySeg);
    rtcIntersect(scene->rtcScene, rtcRay);
    if (rtcRay.geomID == RTC_INVALID_GEOMETRY_ID) {
        return false;
    }

    shapeInst.obj = scene->objects[rtcRay.geomID].get();
    shapeInst.primID = rtcRay.primID;
    return true;
}

bool Occluded(const Scene *scene, const Float time, const Ray &ray, const Float dist) {
    Float minT, maxT;
    if (dist == std::numeric_limits<Float>::infinity()) {
        minT = c_IsectEpsilon;
        maxT = std::numeric_limits<Float>::infinity();
    } else {
        minT = c_IsectEpsilon;
        maxT = (Float(1.0) - c_ShadowEpsilon) * dist;
    }
    return Occluded(scene, time, RaySegment{ray, minT, maxT});
}

bool Occluded(const Scene *scene, const Float time, const RaySegment &raySeg) {
    RTCRay rtcRay = ToRTCRay(time, raySeg);
    rtcOccluded(scene->rtcScene, rtcRay);
    return rtcRay.geomID == 0;
}

const Light *PickLight(const Scene *scene, const Float u, Float &prob) {
    int lightId = scene->lightDist->SampleDiscrete(u, &prob);
    return scene->lights[lightId].get();
}

const Float PickLightProb(const Scene *scene, const Light *light) {
    return light->samplingWeight / scene->lightWeightSum;
}

int GetSceneSerializedSize() {
    return 1 + GetCameraSerializedSize() + GetBSphereSerializedSize();
}

Float *Serialize(const Scene *scene, Float *buffer) {
    buffer = Serialize(BoolToFloat(scene->options->useLightCoordinateSampling), buffer);
    buffer = Serialize(scene->camera.get(), buffer);
    buffer = Serialize(scene->bSphere, buffer);
    return buffer;
}

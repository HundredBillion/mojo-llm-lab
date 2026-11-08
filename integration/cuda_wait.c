#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fail(const char *operation, int status) {
    fprintf(stderr, "LLAMA_CUDA_WAIT error=%s status=%d\n", operation, status);
    _exit(78);
}

__attribute__((constructor)) static void configure_waiting(void) {
    char executable[4096];
    ssize_t size = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
    if (size < 0 || size >= (ssize_t)sizeof(executable) - 1) return;
    executable[size] = 0;
    const char *name = strrchr(executable, '/');
    const char *target = getenv("LLAMA_CUDA_WAIT_TARGET");
    if (strcmp(name ? name + 1 : executable, "llama-server") &&
        (!target || strcmp(executable, target))) return;
    const char *policy = getenv("LLAMA_CUDA_WAIT");
    unsigned int requested = 4;
    if (policy && !strcmp(policy, "auto")) requested = 0;
    else if (policy && strcmp(policy, "blocking")) fail("invalid-policy", 0);

    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!driver) fail("load-driver", 0);
    int (*init)(unsigned int) = dlsym(driver, "cuInit");
    int (*count)(int *) = dlsym(driver, "cuDeviceGetCount");
    int (*get_device)(int *, int) = dlsym(driver, "cuDeviceGet");
    int (*get_state)(int, unsigned int *, int *) = dlsym(driver, "cuDevicePrimaryCtxGetState");
    int (*set_flags)(int, unsigned int) = dlsym(driver, "cuDevicePrimaryCtxSetFlags_v2");
    if (!set_flags) set_flags = dlsym(driver, "cuDevicePrimaryCtxSetFlags");
    if (!init || !count || !get_device || !get_state || !set_flags) fail("driver-symbols", 0);
    int status = init(0);
    if (status) fail("initialize-driver", status);
    int devices = 0;
    if ((status = count(&devices))) fail("enumerate-devices", status);
    if (devices < 1) fail("no-devices", 0);
    for (int ordinal = 0; ordinal < devices; ordinal++) {
        int device = 0, active = 0;
        unsigned int flags = 0;
        if ((status = get_device(&device, ordinal))) fail("get-device", status);
        if ((status = get_state(device, &flags, &active))) fail("read-policy", status);
        if ((flags & 7) != requested) {
            if (active) fail("context-already-active", 0);
            if ((status = set_flags(device, (flags & ~7u) | requested))) fail("set-policy", status);
        }
        if ((status = get_state(device, &flags, &active))) fail("verify-policy", status);
        if ((flags & 7) != requested) fail("policy-readback", 0);
        fprintf(stderr, "LLAMA_CUDA_WAIT device=%d policy=%s flags=%u verified=1\n",
                ordinal, requested == 4 ? "blocking" : "auto", flags);
    }
}

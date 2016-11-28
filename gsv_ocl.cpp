#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#if defined (PLATFORM_LINUX) || defined (PLATFORM_MAC)
#include <unistd.h>
#define Sleep(n) usleep((n)*1000)
#endif

// #define VERBOSE

typedef uint32_t u32;
typedef uint64_t u64;

#include "gsv_ocl.h"

#ifdef PLATFORM_WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

#define Log printf

static int numDevices = -12345;
static ocl_context_t *ocl_context;

const char *ocl_strerror(cl_int status)
{
  switch (status)
  {
    case CL_SUCCESS:                            return "Success!";
    case CL_DEVICE_NOT_FOUND:                   return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:               return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:             return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:                   return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:                 return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:                   return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:              return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:              return "Program build failure";
    case CL_MAP_FAILURE:                        return "Map failure";
    case CL_INVALID_VALUE:                      return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:                return "Invalid device type";
    case CL_INVALID_PLATFORM:                   return "Invalid platform";
    case CL_INVALID_DEVICE:                     return "Invalid device";
    case CL_INVALID_CONTEXT:                    return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:           return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:              return "Invalid command queue";
    case CL_INVALID_HOST_PTR:                   return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:                 return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:                 return "Invalid image size";
    case CL_INVALID_SAMPLER:                    return "Invalid sampler";
    case CL_INVALID_BINARY:                     return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:              return "Invalid build options";
    case CL_INVALID_PROGRAM:                    return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:                return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:          return "Invalid kernel definition";
    case CL_INVALID_KERNEL:                     return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:                  return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:                  return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:                   return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:                return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:             return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:            return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:             return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:              return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:            return "Invalid event wait list";
    case CL_INVALID_EVENT:                      return "Invalid event";
    case CL_INVALID_OPERATION:                  return "Invalid operation";
    case CL_INVALID_GL_OBJECT:                  return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:                return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:                  return "Invalid mip-map level";
    default: return "Unknown";
  }
}

cl_int ocl_diagnose(cl_int result, const char *where, ocl_context_t *cont)
{
  if (result != CL_SUCCESS)
  {
    if (where && cont)
      Log("Error %s on device %u\n", where, cont->clientDeviceNo);
    Log("Error code %d, message: %s\n", result, ocl_strerror(result));
  }

  return result;
}

ocl_context_t *ocl_get_context(int device)
{
  if (device >= 0 && device < numDevices)
    return &ocl_context[device];

  Log("INTERNAL ERROR: bad OpenCL device index %d (detected %d)!\n", device, numDevices);
  return NULL;
}

// To debug on CPU...
// #undef  CL_DEVICE_TYPE_GPU
// #define CL_DEVICE_TYPE_GPU CL_DEVICE_TYPE_ALL

int ocl_initialize_devices(void)
{
  if (numDevices != -12345)
    return numDevices;

  numDevices = -1;  /* assume detection failure for now */

  cl_uint devicesDetected = 0;
  cl_uint numPlatforms;
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS)
  {
    Log("Error obtaining number of platforms (clGetPlatformIDs/1)\n");
    ocl_diagnose(status, NULL, NULL);  // decode error code only
  }
  if (status == CL_SUCCESS)
  {
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;
    char buf_name[256], buf_vendor[256];

    if (numPlatforms != 0)
    {
      // Allocate enough space for each platform
      platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));

      // Fill in platforms with clGetPlatformIDs()
      status = clGetPlatformIDs(numPlatforms, platforms, NULL);
      if (status != CL_SUCCESS)
      {
        Log("Error obtaining list of platforms (clGetPlatformIDs/2)\n");
        ocl_diagnose(status, NULL, NULL);  // decode error code only
      }
      else
      {
        // Use clGetDeviceIDs() to retrieve the number of devices present
        for (cl_uint plat = 0; plat < numPlatforms; plat++)
        {
          cl_uint devcnt;

          if (ocl_diagnose( clGetPlatformInfo(platforms[plat], CL_PLATFORM_NAME, sizeof(buf_name), buf_name, NULL), NULL, NULL ) != CL_SUCCESS)
            *buf_name = 0;
          if (ocl_diagnose( clGetPlatformInfo(platforms[plat], CL_PLATFORM_VENDOR, sizeof(buf_vendor), buf_vendor, NULL), NULL, NULL ) != CL_SUCCESS)
            *buf_vendor = 0;
          Log("Found OCL platform \"%s\" by \"%s\"\n", buf_name, buf_vendor);

          status = clGetDeviceIDs(platforms[plat], CL_DEVICE_TYPE_GPU, 0, NULL, &devcnt);
          if (status == CL_DEVICE_NOT_FOUND)  // Special case. No GPU devices but other may exist
          {
            status = CL_SUCCESS;
            devcnt = 0;
          }
          if (status != CL_SUCCESS)
          {
            Log("Error obtaining number of devices on platform %u (clGetDeviceIDs/1)\n", plat);
            ocl_diagnose(status, NULL, NULL);  // decode error code only
            break;
          }
          Log("  GPU devices on platform: %u\n", devcnt);
          devicesDetected += devcnt;
        }
      }
    }

    if (status == CL_SUCCESS && devicesDetected != 0)
    {
      // Allocate enough space for each device
      devices = (cl_device_id*) malloc(devicesDetected * sizeof(cl_device_id));

      // Allocate and zero space for ocl_context
      ocl_context = (ocl_context_t*) calloc(devicesDetected, sizeof(ocl_context_t));

      // Fill in devices with clGetDeviceIDs()
      cl_uint offset = 0;
      for (cl_uint plat = 0; plat < numPlatforms; plat++)
      {
        cl_uint devcnt;

        if (offset >= devicesDetected)   /* Avoid call with bufferSize=0 for last platform without GPU devices */
          break;

        status = clGetDeviceIDs(platforms[plat], CL_DEVICE_TYPE_GPU, devicesDetected - offset, devices + offset, &devcnt);
        if (status == CL_DEVICE_NOT_FOUND)  // Special case. No GPU devices but other may exist
        {
          status = CL_SUCCESS;
          devcnt = 0;
        }
        if (status != CL_SUCCESS)
        {
          Log("Error obtaining list of devices on platform %u (clGetDeviceIDs/2)\n", plat);
          ocl_diagnose(status, NULL, NULL);  // decode error code only
          break;
        }

        // Fill non-zero context fields for each device
        for (cl_uint u = 0; u < devcnt; u++, offset++)
        {
          ocl_context_t *cont = &ocl_context[offset];

          /* Assume it working for now */
          cont->active            = true;
          cont->coreID            = CORE_NONE;
          cont->platformID        = platforms[plat];
          cont->deviceID          = devices[offset];
          cont->firstOnPlatform   = (u == 0);
          cont->clientDeviceNo    = offset;
          cont->runSize           = 65536;
          cont->runSizeMultiplier = 64;
          cont->maxWorkSize       = 2048 * 2048;

          status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_NAME, sizeof(buf_name), buf_name, NULL);
          if (ocl_diagnose(status, "clGetDeviceInfo(CL_DEVICE_NAME)", cont) != CL_SUCCESS)
            *buf_name = 0;
          Log("  D%u: \"%s\"\n", offset, buf_name);

          /* Sanity check: size_t must be same width for both client and device */
          /* Seems to be too paranoid and useless, so only write a message */
          cl_uint devbits;
          status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_ADDRESS_BITS, sizeof(devbits), &devbits, NULL);
#if 0
          if (ocl_diagnose(status, "clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS)", cont) != CL_SUCCESS)
            cont->active = false;
          else if (devbits != sizeof(size_t) * 8)
          {
            if (sizeof(size_t) * 8 > devbits)  // Host 64, device 32 - use workarounds
              Log("Warning: Bitness of device %u (%u) does not match CPU (%u), will try to work around\n", offset, devbits, (unsigned)(sizeof(size_t) * 8));
            else
            {
              Log("Error: Bitness of device %u (%u) does not match CPU (%u)!\n", offset, devbits, (unsigned)(sizeof(size_t) * 8));
              cont->active = false;
            }
          }
#else
          if (ocl_diagnose(status, "clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS)", cont) == CL_SUCCESS && devbits != sizeof(size_t) * 8)
            Log("size_t on device %u: %u bits, host: %u bits\n", offset, devbits, (unsigned)(sizeof(size_t) * 8));
#endif
        }
      }
    }

    if (status == CL_SUCCESS)
    {
      // Everything is done. Apply configuration.
      numDevices = devicesDetected;
    }

    // Don't need them anymore
    if (devices)
      free(devices);
    if (platforms)
      free(platforms);
  }

  return numDevices;
}

void ocl_cleanup_device(ocl_context_t *cont, bool full_cleanup)
{
  cont->coreID = CORE_NONE;

  if (cont->kernel)
  {
    ocl_diagnose( clReleaseKernel(cont->kernel), "clReleaseKernel", cont );
    cont->kernel = NULL;
  }
  
  if (cont->program)
  {
    ocl_diagnose( clReleaseProgram(cont->program), "clReleaseProgram", cont );
    cont->program = NULL;
  }

#if 0
  if (cont->const_buffer)
  {
    ocl_diagnose( clReleaseMemObject(cont->const_buffer), "clReleaseMemObject(const_buffer)", cont );
    cont->const_buffer = NULL;
  }

  if (cont->out_buffer)
  {
    ocl_diagnose( clReleaseMemObject(cont->out_buffer),  "clReleaseMemObject(out_buffer)", cont );
    cont->out_buffer = NULL;
  }
#endif

  if (full_cleanup)
  {
    if (cont->cmdQueue)
    {
      ocl_diagnose( clReleaseCommandQueue(cont->cmdQueue), "clReleaseCommandQueue", cont );
      cont->cmdQueue = NULL;
    }

    if (cont->clcontext)
    {
      ocl_diagnose( clReleaseContext(cont->clcontext), "clReleaseContext", cont );
      cont->clcontext = NULL;
    }

    cont->runSize = 65536;
    cont->maxWorkSize = 2048 * 2048;
  }
}

static unsigned g_pipes_count;

static char *try_load_core(const char *coreformat, unsigned core, unsigned pipes)
{
  char corefile[64];
  char *buf = NULL;
  FILE *f;

  sprintf(corefile, coreformat, core);
  f = fopen(corefile, "rt");
  if (f)
  {
    int flen = fseek(f, 0, SEEK_END);   // filelength(fileno(f)) not supported on Linux
    if (flen > 0 && (buf = (char*)malloc(flen + 1)) != NULL)
    {
      fseek(f, 0, SEEK_SET);
      flen = fread(buf, flen, 1, f);  // could return less then requested due to CR/LF translation on Windows
      buf[flen] = 0;
      g_pipes_count = pipes;
    }
    fclose(f);
  }
  return buf;
}

static cl_int ocl_build_program(ocl_context_t *cont, const char* programText, const char *kernelName, const char *options, unsigned core)
{
  cl_int status;

  char *buf;
  g_pipes_count = 1;
  buf = try_load_core("ocl_core_%u.cl", core, 1);
  if (buf == NULL)
    buf = try_load_core("ocl_core_%u_p2.cl", core, 2);
  if (buf == NULL)
    buf = try_load_core("ocl_core_%u_p4.cl", core, 4);
  if (buf)
    programText = buf;

  cont->program = clCreateProgramWithSource(cont->clcontext, 1, &programText, NULL, &status);
  if (ocl_diagnose(status, "clCreateProgramWithSource", cont) != CL_SUCCESS)
    return status;

  if (buf)
    free(buf);

  status = clBuildProgram(cont->program, 1, &cont->deviceID, options, NULL, NULL);
  // status = clBuildProgram(cont->program, 1, &cont->deviceID, "-cl-std=CL1.1", NULL, NULL);
  ocl_diagnose(status, "building cl program", cont);

  size_t log_size;
  cl_int temp;

  temp = clGetProgramBuildInfo(cont->program, cont->deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if (ocl_diagnose(temp, "clGetProgramBuildInfo", cont) == CL_SUCCESS)
  {
    char *buf;

    if (log_size && (buf = (char *) malloc(log_size+1)) != NULL)
    {
      temp = clGetProgramBuildInfo(cont->program, cont->deviceID, CL_PROGRAM_BUILD_LOG, log_size, buf, NULL);
      ocl_diagnose(temp, "clGetProgramBuildInfo", cont);
      buf[log_size] = '\0';
      /* Check if log contains something (not only whitespace) */
      for (char *p = buf; *p; p++)
      {
        if (!isspace(*p))
        {
          Log("Build Log (%u bytes):\n", (unsigned)log_size);
          Log("%s\n", buf);
          break;
        }
      }
      free(buf);
    }
  }
  if (status != CL_SUCCESS)
    return status;

  cont->kernel = clCreateKernel(cont->program, kernelName, &status);
  if (ocl_diagnose(status, "building kernel", cont) != CL_SUCCESS)
    return status;

  /* Get a performance hint */
  size_t prefm;
  status = clGetKernelWorkGroupInfo(cont->kernel, cont->deviceID, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                    sizeof(prefm), &prefm, NULL);
  if (ocl_diagnose(status, "clGetKernelWorkGroupInfo", cont) == CL_SUCCESS)
  {
    size_t cus;
    status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
    if (ocl_diagnose(status, "clGetDeviceInfo(MAX_COMPUTE_UNITS)", cont) == CL_SUCCESS)
    {
      cont->runSizeMultiplier = prefm * cus /* * 4 */; //Hack for now. We need 4 wavefronts per CU to hide latency
#ifdef VERBOSE
      Log("ocl_runSizeMultiplier: 0x%04X\n", cont->runSizeMultiplier);
#endif
    }
  }

#ifdef VERBOSE
  cl_ulong constmem;
  if (clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(constmem), &constmem, NULL) == CL_SUCCESS)
    Log("CL_MAX_CONSTANT_BUFFER_SIZE: %uK\n", (unsigned)(constmem / 1024));
#endif

  return CL_SUCCESS;
}

/*
 * Setup basic parameters like context and command queue, which are
 * not changing in this app even if core was changed.
 *
 * Note: device must be in "full clean' state.
 */
cl_int ocl_preconfigure(ocl_context_t *cont)
{
  cl_int status;

  cont->clcontext = clCreateContext(NULL, 1, &cont->deviceID, NULL, NULL, &status);
  if (ocl_diagnose(status, "creating OCL context", cont) != CL_SUCCESS)
    return status;

  cont->cmdQueue = clCreateCommandQueue(cont->clcontext, cont->deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
  if (ocl_diagnose(status, "creating command queue", cont) != CL_SUCCESS)
    return status;

  return CL_SUCCESS;
}

static const char unknown_program[] = "#error Internal error: unknown core requested\n";

static const char core_source_63[] =
#include "core_63.tmp"
;

static const char core_source_64[] =
#include "core_64.tmp"
;

static const char core_source_79[] =
#include "core_79.tmp"
;

cl_int ocl_execute_core(u32 core, ocl_context_t *cont, cl_event *pEvent, u64 *pQueuedTime, u32 iterations,
                        cl_mem fac_mult_ratio,  cl_mem init, u32 bmax, u32 count, cl_mem RES, u32 init_fac_shift,
                        cl_mem fac_mult_ratio1, cl_mem init1)
{
  cl_int status;

  if (cont->coreID != core)
  {
    /* Kernel change in progress, finish all pending operations */
    status = ocl_diagnose( clFinish(cont->cmdQueue), "clFinish", cont );
    if (status != CL_SUCCESS)
      return status;

    ocl_cleanup_device(cont, false);

    /*
     * Warning: using a fact that all parameters (even buffers) are constant.
     * So numeric constant are passed as #defines in source code and
     * all buffers are mapped only once.
     */
    const char *program, *entry;
    char options[256];
    sprintf(options, "-Dbmax=%u -Dparam_count=%u -Dinit_fac_shift=%u", bmax, count, init_fac_shift);
    switch (core)
    {
      case 63: program = core_source_63;  entry = "process63"; break;
      case 64: program = core_source_64;  entry = "process64"; break;
      case 79: program = core_source_79;  entry = "process79"; break;
      default: program = unknown_program; entry = "fake_entry"; break;
    }
    status = ocl_build_program(cont, program, entry, options, core);
    if (status != CL_SUCCESS)
      return status;

    status = clSetKernelArg(cont->kernel, 0, sizeof(fac_mult_ratio), &fac_mult_ratio);
    if (status == CL_SUCCESS)
      status = clSetKernelArg(cont->kernel, 1, sizeof(init), &init);
    if (status == CL_SUCCESS)
      status = clSetKernelArg(cont->kernel, 2, sizeof(RES), &RES);
    /* Two parameters are optional (used only in some cores) */
    if (status == CL_SUCCESS && fac_mult_ratio1)
    {
      status = clSetKernelArg(cont->kernel, 3, sizeof(fac_mult_ratio1), &fac_mult_ratio1);
      if (status == CL_SUCCESS)
        status = clSetKernelArg(cont->kernel, 4, sizeof(init1), &init1);
    }
    if (ocl_diagnose(status, "setting kernel args", cont) != CL_SUCCESS)
      return status;

#ifdef VERBOSE
    Log("Iterations requested: 0x%X\n", iterations);

    size_t memsiz;
    if (clGetMemObjectInfo(fac_mult_ratio, CL_MEM_SIZE, sizeof(memsiz), &memsiz, NULL) == CL_SUCCESS)
      Log(" fac_mult_ratio: %uK\n", (unsigned)(memsiz / 1024));
    if (clGetMemObjectInfo(init, CL_MEM_SIZE, sizeof(memsiz), &memsiz, NULL) == CL_SUCCESS)
      Log("           init: %uK\n", (unsigned)(memsiz / 1024));
    if (clGetMemObjectInfo(RES, CL_MEM_SIZE, sizeof(memsiz), &memsiz, NULL) == CL_SUCCESS)
      Log("            RES: %uK\n", (unsigned)(memsiz / 1024));
    if (fac_mult_ratio1)
    {
      if (clGetMemObjectInfo(fac_mult_ratio1, CL_MEM_SIZE, sizeof(memsiz), &memsiz, NULL) == CL_SUCCESS)
        Log("fac_mult_ratio1: %uK\n", (unsigned)(memsiz / 1024));
      if (clGetMemObjectInfo(init1, CL_MEM_SIZE, sizeof(memsiz), &memsiz, NULL) == CL_SUCCESS)
        Log("          init1: %uK\n", (unsigned)(memsiz / 1024));
    }
#endif

    cont->coreID = core;
  }

  size_t globalWorkSize[1];
  globalWorkSize[0] = iterations / g_pipes_count;
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, pEvent);
  if (ocl_diagnose(status, "clEnqueueNDRangeKernel", cont) != CL_SUCCESS)
    return status;

#if 0 // !!! Disabled because result is currently not used in nvidia_wait_for_event
  /* If requested, return system tick (transparent units) when command was queued */
  if (pQueuedTime)
  {
    LARGE_INTEGER Now;
    QueryPerformanceCounter(&Now);
    *pQueuedTime = Now.QuadPart;
  }
#else
  (void) pQueuedTime;
#endif

  status = clFlush(cont->cmdQueue);
  if (ocl_diagnose(status, "clFlush", cont) != CL_SUCCESS)
    return status;

  return CL_SUCCESS;
}

cl_int ocl_nvidia_wait_for_event(u32 mode, cl_event event, u64 last_kernel_time, u64 queued_systick, ocl_context_t *cont)
{
/* Unfortunately, I cannot make following simple code to work.
   For few first seconds, it uses 100% CPU, then CPU usage drops but
   GPU performance becomes very poor (<50%) and finally it hangs on Ctrl-C.

   cl_int ev_status;
   ResetEvent(gd.ocl_system_event);
   CUDA_error_exit( ocl_diagnose( clSetEventCallback(event_in_progress, CL_COMPLETE, ocl_event_callback, NULL), "clSetEventCallback", gd.device_ctx ), __LINE__ );
   CUDA_error_exit( ocl_diagnose( clGetEventInfo(event_in_progress, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ev_status), &ev_status, NULL), "clGetEventInfo", gd.device_ctx ), __LINE__ );
   if (ev_status < CL_COMPLETE)
      CUDA_error_exit(ev_status, __LINE__);
   if (ev_status != CL_COMPLETE)
      WaitForSingleObject(gd.ocl_system_event, INFINITE);

(ocl_event_callback function will do SetEvent(gd.ocl_system_event))
*/

/*
 * An idea behind this workaround is to sleep as close as possible to the average kernel
 * execution time (which is more or less constant) so NVIDIA busy-loop will take small
 * amount of time, less then 1ms. CPU usage of 5-10% is much better then 100%.
 */
  cl_int status;
  cl_int reqd_event_status;

#if 0
  switch (mode)
  {
  case 1: // smart sleep
    reqd_event_status = CL_RUNNING;
    break;
  case 2:
    reqd_event_status = CL_COMPLETE; // just wait for completion
    break;
  default:
    return CL_SUCCESS;
  }
#else
  /* See comment below */
  (void) mode;
  (void) last_kernel_time;
  (void) queued_systick;     // !!! also disabled in execute_core - be sure to re-enable it there if used
  reqd_event_status = CL_COMPLETE;
#endif

  /* Wait for kernel to be in required state */
  for (;;)
  {
    cl_int ev_state;

    status = ocl_diagnose( clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(ev_state), &ev_state, NULL), "clGetEventInfo", cont );
    if (status != CL_SUCCESS)
      return status;
    if (ev_state == CL_COMPLETE)  // done, no need to wait
      return CL_SUCCESS;
    if (ev_state < CL_COMPLETE)  // negative is an error code
    {
      ocl_diagnose(ev_state, "ev_state", cont);
      return ev_state;
    }
    if (ev_state <= reqd_event_status)
      break;
#if 0
  static u32 counter1;
  if ((++counter1 & 63) == 0)
    printf("\nEvent state: %d\n", (int)ev_state);
#endif
    Sleep(1);
  }

/*
 * Alas, I wrote this shining piece of code just to find out that NVIDIA crap never
 * gets info CL_RUNNING state! It jumps into CL_COMPLETE immediately from CL_SUBMITTED.
 * So I disabled everything and reduced the function to simple Sleep(1) until event is complete.
 * With high-resoluion 1ms system timer and high 'B' parameter, it works quite well.
 */

#if 0
  /* Now get OCL timestamps for "Queued" and "Running" points */
  cl_ulong queuedOclTime, startOclTime;

  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedOclTime, 0);
  if (ocl_diagnose(status, "clGetEventProfilingInfo(COMMAND_QUEUED)", cont) != CL_SUCCESS)
    return status;
  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startOclTime, 0);
  if (ocl_diagnose(status, "clGetEventProfilingInfo(COMMAND_START)", cont) != CL_SUCCESS)
    return status;

  /* Now we have delay between queueing and execution, along with expected kernel runtime (which is still running) */
  u64 expected_run_time_usec = (startOclTime - queuedOclTime + last_kernel_time) / 1000;  // nanosec to usec

  /* Find system time passed since queueing */
  static LARGE_INTEGER Frequency;
  if (Frequency.QuadPart == 0)
    QueryPerformanceFrequency(&Frequency);

  LARGE_INTEGER Now;
  QueryPerformanceCounter(&Now);

  u64 passed_usec = (Now.QuadPart - queued_systick) * 1000000 / Frequency.QuadPart;

#if 1
  static u32 counter;
  if ((++counter & 63) == 0)
    printf("\npassed_usec %u, expected_run_time_usec %u\n", (unsigned)passed_usec, (unsigned)expected_run_time_usec);
#endif

  /* should we sleep? */
  if (passed_usec < expected_run_time_usec)
  {
    u64 sleep_usec = expected_run_time_usec - passed_usec;
    if (sleep_usec >= 3000)
      Sleep((u32)(sleep_usec / 1000)-2);
  }
#endif

  return CL_SUCCESS;
}

/*
 * Return time used by kernel in OpenCL ticks (nanoseconds)
 */
cl_int ocl_get_kernel_exec_time(cl_event event, u64 *pTime, ocl_context_t *cont)
{
  cl_int status;
  cl_ulong startTime, endTime;

  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, 0);
  if (ocl_diagnose(status, "clGetEventProfilingInfo(COMMAND_START)", cont) != CL_SUCCESS)
    return status;

  status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, 0);
  if (ocl_diagnose(status, "clGetEventProfilingInfo(COMMAND_END)", cont) != CL_SUCCESS)
    return status;

  *pTime = endTime - startTime;
  return CL_SUCCESS;
}

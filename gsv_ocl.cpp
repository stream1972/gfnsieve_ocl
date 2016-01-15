#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint32_t u32;

#include "gsv_ocl.h"

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
   #undef  CL_DEVICE_TYPE_GPU
   #define CL_DEVICE_TYPE_GPU CL_DEVICE_TYPE_ALL

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

          /* Sanity check: size_t must be same width for both client and device */
          cl_uint devbits;
          status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_ADDRESS_BITS, sizeof(devbits), &devbits, NULL);
          if (ocl_diagnose(status, "clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS)", cont) != CL_SUCCESS)
            cont->active = false;
          else if (devbits != sizeof(size_t) * 8)
          {
            Log("Error: Bitness of device %u (%u) does not match CPU (%u)!\n", u, devbits, sizeof(size_t) * 8);
            cont->active = false;
          }
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

static cl_int ocl_build_program(ocl_context_t *cont, const char* programText, const char *kernelName, const char *options)
{
  cl_int status;

  cont->program = clCreateProgramWithSource(cont->clcontext, 1, &programText, NULL, &status);
  if (ocl_diagnose(status, "clCreateProgramWithSource", cont) != CL_SUCCESS)
    return status;

  status = clBuildProgram(cont->program, 1, &cont->deviceID, options, NULL, NULL);
  // status = clBuildProgram(cont->program, 1, &cont->deviceID, "-cl-std=CL1.1", NULL, NULL);
  ocl_diagnose(status, "building cl program", cont);

  size_t log_size;
  cl_int temp;

  temp = clGetProgramBuildInfo(cont->program, cont->deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if (ocl_diagnose(temp, "clGetProgramBuildInfo", cont) == CL_SUCCESS)
  {
    char *buf;

    Log("Build Log (%d bytes):\n", log_size);
    if (log_size && (buf = (char *) malloc(log_size)) != NULL)
    {
      temp = clGetProgramBuildInfo(cont->program, cont->deviceID, CL_PROGRAM_BUILD_LOG, log_size, buf, NULL);
      ocl_diagnose(temp, "clGetProgramBuildInfo", cont);
      buf[log_size-1] = '\0';
//    for (unsigned i = 0; i < log_size && isspace(buf[i]); i++)
//      ;
      Log("%s\n", buf);
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
      Log("ocl_runSizeMultiplier set to 0x%X\n", cont->runSizeMultiplier);
    }
  }

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

#define MSTRINGIFY(A) #A
static const char core_source_63[] = 
#include "ocl_core_63.cl"

cl_int ocl_execute_core(u32 core, ocl_context_t *cont, cl_event *pEvent, u32 iterations,
                        cl_mem fac_mult_ratio,  cl_mem init, u32 bmax, u32 count, cl_mem RES, u32 init_fac_shift,
                        cl_mem fac_mult_ratio1, cl_mem init1)
{
  cl_int status;

  if (cont->coreID != core)
  {
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
      default: program = unknown_program; entry = "fake_entry"; break;
    }
    status = ocl_build_program(cont, program, entry, options);
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

    Log("Iterations requested: 0x%X\n", iterations);

    cont->coreID = core;
  }

  size_t globalWorkSize[1];
  globalWorkSize[0] = iterations;
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, pEvent);
  if (ocl_diagnose(status, "clEnqueueNDRangeKernel", cont) != CL_SUCCESS)
    return status;

  status = clFlush(cont->cmdQueue);
  if (ocl_diagnose(status, "clFlush", cont) != CL_SUCCESS)
    return status;

  return CL_SUCCESS;
}

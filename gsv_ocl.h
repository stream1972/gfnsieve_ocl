#include <CL/cl.h>

enum
{
  CORE_NONE
};

typedef struct {
  bool              active;          // structure filled, device passed sanity checks
  bool              firstOnPlatform; // new platform started here (for logs)
  cl_platform_id    platformID;      // in OpenCL subsystem
  cl_device_id      deviceID;        // in OpenCL subsystem
  int               clientDeviceNo;  // client GPU index (for logs)
  cl_context        clcontext;
  cl_command_queue  cmdQueue; 
#if 0
  cl_mem            const_buffer; 
  cl_mem            out_buffer; 
#endif
  u32               coreID;
  cl_program	    program; 
  cl_kernel         kernel; 
  u32               runSize;
  u32               runSizeMultiplier;
  u32               maxWorkSize;
} ocl_context_t;

int            ocl_initialize_devices(void);
void           ocl_cleanup_device(ocl_context_t *cont, bool full_cleanup);
ocl_context_t *ocl_get_context(int device);
cl_int         ocl_diagnose(cl_int result, const char *where, ocl_context_t *cont);
const char    *ocl_strerror(cl_int status);
cl_int         ocl_preconfigure(ocl_context_t *cont);
cl_int         ocl_execute_core(u32 core, ocl_context_t *cont, cl_event *pEvent, u32 iterations,
                                cl_mem fac_mult_ratio,  cl_mem init, u32 bmax, u32 count, cl_mem RES, u32 init_fac_shift,
                                cl_mem fac_mult_ratio1, cl_mem init1);
"""Darwin Mach memory observation without subprocesses or new dependencies.

Apple vm_statistics/host_info ABI: HOST_VM_INFO (2), 15 natural_t fields.
Admission uses free+inactive pages as a reclaimability estimate, not a promise
of allocatable bytes. Speculative pages already belong to free_count; purgeable
pages are not added because counts can overlap. Failures never become capacity.
"""
import ctypes
import sys


def observe(*,library=None):
    if sys.platform!='darwin':raise ValueError('Darwin_memory_observation_required')
    lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib') if library is None else library
    lib.mach_host_self.argtypes=[];lib.mach_host_self.restype=ctypes.c_uint
    lib.host_page_size.argtypes=[ctypes.c_uint,ctypes.POINTER(ctypes.c_size_t)];lib.host_page_size.restype=ctypes.c_int
    lib.host_statistics.argtypes=[ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_uint),ctypes.POINTER(ctypes.c_uint)];lib.host_statistics.restype=ctypes.c_int
    lib.mach_port_deallocate.argtypes=[ctypes.c_uint,ctypes.c_uint];lib.mach_port_deallocate.restype=ctypes.c_int
    host=lib.mach_host_self()
    if not host:raise OSError('mach_host_self_failed')
    failure=None
    try:
        page=ctypes.c_size_t();count=ctypes.c_uint(15);values=(ctypes.c_uint*15)()
        status=lib.host_page_size(host,ctypes.byref(page))
        if status or page.value not in (4096,16384,65536):raise OSError('host_page_size_failed: '+str(status))
        status=lib.host_statistics(host,2,values,ctypes.byref(count))
        if status or count.value!=15:raise OSError('host_statistics_failed: '+str(status))
        return dict(page_size=page.value,free_pages=values[0],inactive_pages=values[2],
            admission_bytes=(values[0]+values[2])*page.value,basis='free_plus_inactive_estimate')
    except BaseException as exc:
        failure=exc
        raise
    finally:
        task=ctypes.c_uint.in_dll(lib,'mach_task_self_').value if library is None else lib.task_self
        if lib.mach_port_deallocate(task,host):
            if failure is not None:failure.add_note('mach_host_port_release_failed')
            else:raise OSError('mach_host_port_release_failed')

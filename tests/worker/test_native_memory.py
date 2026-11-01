"""Mach memory ABI/port custody and fail-closed admission, without subprocesses."""
import ctypes
import unittest
from unittest.mock import Mock,patch
from worker.native_memory import observe


class NativeMemoryTests(unittest.TestCase):
    def library(self):
        lib=Mock();lib.task_self=31
        lib.mach_host_self.return_value=42;lib.mach_port_deallocate.return_value=0
        def pages(host,out):
            self.assertEqual(host,42)
            ctypes.cast(out,ctypes.POINTER(ctypes.c_size_t))[0]=16384
            return 0
        def statistics(host,flavor,values,count):
            self.assertEqual((host,flavor),(42,2))
            self.assertEqual(ctypes.cast(count,ctypes.POINTER(ctypes.c_uint))[0],15)
            values[0]=10;values[2]=20;values[14]=999  # never add overlapping counts
            return 0
        lib.host_page_size.side_effect=pages;lib.host_statistics.side_effect=statistics
        return lib

    def test_correct_abi_units_target_port_and_release(self):
        lib=self.library()
        with patch('sys.platform','darwin'):value=observe(library=lib)
        self.assertEqual(value['admission_bytes'],30*16384)
        self.assertEqual(value['basis'],'free_plus_inactive_estimate')
        lib.mach_port_deallocate.assert_called_once_with(31,42)

    def test_error_and_port_release_failures_cannot_supply_capacity(self):
        for failure in ('page','statistics','count','release'):
            lib=self.library()
            if failure=='page':lib.host_page_size.side_effect=None;lib.host_page_size.return_value=5
            if failure=='statistics':lib.host_statistics.side_effect=None;lib.host_statistics.return_value=5
            if failure=='count':
                def count(host,flavor,values,out):
                    ctypes.cast(out,ctypes.POINTER(ctypes.c_uint))[0]=14
                    return 0
                lib.host_statistics.side_effect=count
            if failure=='release':lib.mach_port_deallocate.return_value=5
            with self.subTest(failure=failure),patch('sys.platform','darwin'),self.assertRaises(OSError):observe(library=lib)
            lib.mach_port_deallocate.assert_called_once_with(31,42)

    def test_non_darwin_never_loads_darwin_symbols_and_admission_fails_closed(self):
        with patch('sys.platform','linux'),patch('ctypes.CDLL') as load:
            with self.assertRaises(ValueError):observe()
        load.assert_not_called()
        from worker.native_supervisor import available_ram
        with patch('sys.platform','darwin'),patch('worker.native_memory.observe',side_effect=OSError('observation failed')):
            self.assertEqual(available_ram(),0)


if __name__=='__main__':unittest.main()

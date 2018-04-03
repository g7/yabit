Yet Another BootImage Tool (`yabit`)
====================================

`yabit` is a python written, device tree-aware tool to create, extract
and update Android BootImages ("boot.img").
It depends only on a reasonably up-to-date Python interpreter (3.4+) and
the standard library.

Usage
-----

	usage: yabit.py [-h] [--debug] [--input INPUT] [--output OUTPUT]
					[--dump {nothing,everything,header,kernel,initramfs,dtbs} [{nothing,everything,header,kernel,initramfs,dtbs} ...]]
					[--dump-to DUMP_TO] [--print-header] [--cmdline CMDLINE]
					[--remove-original-dtbs] [--dtb DTB [DTB ...]]
					[--kernel KERNEL] [--initramfs INITRAMFS] [--base BASE]
					[--kernel-offset KERNEL_OFFSET]
					[--initramfs-offset INITRAMFS_OFFSET]
					[--second-offset SECOND_OFFSET] [--tags-offset TAGS_OFFSET]
					[--pagesize {2048,4096,8192,16384}]

	optional arguments:
	  -h, --help            show this help message and exit
	  --debug               enables debug logging
	  --input INPUT, -i INPUT
							the input image path
	  --output OUTPUT, -o OUTPUT
							the path where to store the output
	  --dump {nothing,everything,header,kernel,initramfs,dtbs} [{nothing,everything,header,kernel,initramfs,dtbs} ...], -u {nothing,everything,header,kernel,initramfs,dtbs} [{nothing,everything,header,kernel,initramfs,dtbs} ...]
							what to dump. Might be specified multiple times.
							Defaults to nothing
	  --dump-to DUMP_TO, -t DUMP_TO
							where to dump the specified images. Defaults to the
							current directory
	  --print-header, -e    prints the header
	  --cmdline CMDLINE, -c CMDLINE
							use the supplied cmdline
	  --remove-original-dtbs, -r
							removes the original dtbs found in the kernel
	  --dtb DTB [DTB ...]   appends the specified dtb to the kernel. It might be
							used multiple times
	  --kernel KERNEL       uses the specified kernel image
	  --initramfs INITRAMFS
							uses the specified initramfs image
	  --base BASE           the base address to use
	  --kernel-offset KERNEL_OFFSET
							the kernel offset
	  --initramfs-offset INITRAMFS_OFFSET
							the initramfs offset
	  --second-offset SECOND_OFFSET
							the second bootloader offset
	  --tags-offset TAGS_OFFSET
							the tags offset
	  --pagesize {2048,4096,8192,16384}, -p {2048,4096,8192,16384}
							the page size to use

Examples
--------

### Replacing the kernel image with another

	./yabit.py --input boot.img --kernel bzImage --output boot.img

### Extracting every component from an image

	./yabit.py --input boot.img --dump everything --dump-to result/

### Replacing eventual compiled DeviceTrees (DTBs) with a custom one

	./yabit.py --input boot.img --remove-original-dtbs --dtb custom_dtb.dtb --output new_boot.img

The `--remove-original-dtbs` switch removes eventual DTBs already present
in the kernel. The specified DTB is appended afterwards.

### Creating a new boot.img

	./yabit.py --kernel bzImage --initramfs initramfs.gz --cmdline "lpm_levels.sleep_disabled=1 user_debug=31 androidboot.selinux=permissive msm_rtb.filter=0x3F ehci-hcd.park=3 dwc3.maximum_speed=high dwc3_msm.prop_chg_detect=Y coherent_pool=8M sched_enable_power_aware=1 androidboot.hardware=kugo" --base 0x80000000 --output boot.img

Notes
-----

`yabit` was written to satisfy a specific need (replacing DeviceTrees on an already
made image) and it grew up to become a sort-of swiss army knife for boot images.

There is still some work to do, though, such as support for the second bootloader
image, better error-handling and the code is not as well commented as I'd like.

Contributions (including extensive testing) are welcome.

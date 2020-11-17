#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# yabit - Yet Another BootImage Tool
# Copyright (C) 2018 Eugenio "g7" Paolantonio
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the <organization> nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
yabit is a python written, device tree-aware tool to create, extract and
update Android BootImages.
"""

import enum
import struct

import argparse

import sys

import os

import io

import hashlib

import traceback

import logging

from collections import namedtuple, OrderedDict

DEFAULT_BASE = 0x10000000

DEFAULT_KERNEL_OFFSET = 0x00008000

DEFAULT_INITRAMFS_OFFSET = 0x01000000

DEFAULT_SECOND_IMAGE_OFFSET = 0x00f00000

DEFAULT_TAGS_OFFSET = 0x00000100

DEFAULT_PAGE_SIZE = 2048

PAGES_TO_READ = 4

# Set-up logging
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class ParseResult(namedtuple("_ParseResult", ["status", "start", "end"])):

	def __new__(cls, *args, **kwargs):

		kwargs.setdefault("start", 0)
		kwargs.setdefault("end", 0)

		return super().__new__(cls, *args, **kwargs)

class StopReason(enum.IntEnum):

	STRUCT_END = 0x01

	WORD = 0x02

	END = 0x04

	SIZE = 0x08

class ParseStatus(enum.IntEnum):

	FOUND = 1

	NOT_FOUND = 2

	MAGIC_WORD_FOUND = 3

class BlockType(enum.IntEnum):

	STRUCT = 1

	NORMAL = 2

class DumpAction(enum.Enum):

	NOTHING = "nothing"

	EVERYTHING = "everything"

	HEADER = "header"

	KERNEL = "kernel"

	INITRAMFS = "initramfs"

	DTBS = "dtbs"

	#SECOND_IMAGE = "second_image"

	def __str__(self):
		"""
		Required for argparse
		"""

		return self.value

FakeStructEnum = namedtuple("FakeStructEnum", ["name", "value"])

class StructEnum(enum.Enum):

	PAD = "x"

	CHAR = "c" # 1
	SIGNED_CHAR = "b" # 1
	UNSIGNED_CHAR = "B" # 1

	BOOLEAN = "?" # 1

	SHORT = "h" # 2
	UNSIGNED_SHORT = "H" # 2

	INTEGER = "i" # 4
	UNSIGNED_INTEGER = "I" # 4

	LONG = "l" # 4
	UNSIGNED_LONG = "L" # 4
	LONG_LONG = "q" # 8
	UNSIGNED_LONG_LONG = "Q" # 8

	SSIZE_T = "n"
	SIZE_T = "N"

	FLOAT = "f" # 4
	DOUBLE = "d" # 8

	STRING = "s"
	PASCAL_STRING = "p"

	def __mul__(self, factor):
		"""
		Handles multiplications.
		"""

		return FakeStructEnum(
			name=self.name,
			value="%d%s" % (factor, self.value)
		)

class BaseBlock:

	"""
	A searchable block of bytes defined by a magic word.
	"""

	MAGIC_WORD = None

	PARAMS = {
		"content" : None
	}

	DEFAULTS = {}

	BLOCK_TYPE = BlockType.NORMAL

	CONTENT_PARAM = "content"

	STOP_REASON = StopReason.END

	STOP_WORD = b""

	def __init__(self, page_size=DEFAULT_PAGE_SIZE):
		"""
		Initializes the class.

		:param: page_size: the page size to use (defaults to DEFAULT_PAGE_SIZE)
		"""

		self.page_size = page_size

		self.content = {}
		self.content.update(self.DEFAULTS)

		self.found = False

	@property
	def size(self):
		"""
		Returns the full size of the block.
		"""

		if self.BLOCK_TYPE == BlockType.NORMAL and self.CONTENT_PARAM in self.content:
			return len(self.content[self.CONTENT_PARAM])

		return 0

	def __getitem__(self, item):
		"""
		Returns the item from the content dictionary, or padding bytes
		if it doesn't exist.
		"""

		if item in self.content:
			return self.content[item]

		if not item in self.PARAMS:
			raise AttributeError("%s not in PARAMS" % item)

		return None

	def __contains__(self, item):
		"""
		Forwards the query to the inner content dictionary.
		"""

		return self.content.__contains__(item)

	def parse(self, chunk, is_end, current_size=0):
		"""
		Parses a chunk of bytes.

		:param: chunk: the chunk to parse
		:param: last_chunk: the last chunk, for context (or b"")
		:param: is_end: True if the file reached its end, False if not
		:returns: if successful, returns the bytes to seek to position
		the cursor just at the end of the block. In case of failure, it
		returns -1.
		"""

		logger.debug("Current size is %d" % current_size)

		if current_size >= self.size and self.STOP_REASON & StopReason.SIZE:
			logger.debug("Handling StopReason.SIZE")
			return ParseResult(status=ParseStatus.FOUND, end=(len(chunk) - (current_size - self.size)))
		elif not is_end and self.STOP_REASON & StopReason.WORD:
			logger.debug("Handling StopReason.WORD")
			if self.MAGIC_WORD is not None:
				magic_start = chunk.find(self.MAGIC_WORD, 0)
				magic_end = chunk.find(self.STOP_WORD, 1 if self.STOP_WORD == self.MAGIC_WORD else 0)

				if magic_start >= 0 and (magic_end - magic_start) > 0:
					# We both have a start and an end
					self.content["content"] = chunk[magic_start:magic_end]

					# Position ourselves at the end of the block
					return ParseResult(status=ParseStatus.FOUND, start=magic_start, end=magic_end)
				elif magic_start >= 0:
					# Got a start
					return ParseResult(status=ParseStatus.MAGIC_WORD_FOUND, start=magic_start)
			else:
				# Assume STOP_WORD is defined
				magic_end = chunk.find(self.STOP_WORD, 0)

				if magic_end >= 0:
					# Got it!
					return ParseResult(status=ParseStatus.FOUND, end=magic_end)
		elif is_end and self.STOP_REASON & StopReason.END:
			# We reached the end, so we possibly got the whole
			# block
			logger.debug("Handling StopReason.END")
			magic_start = chunk.find(self.MAGIC_WORD) if self.MAGIC_WORD is not None else 0

			if magic_start >= 0:
				return ParseResult(status=ParseStatus.FOUND, start=magic_start, end=len(chunk))

		return ParseResult(status=ParseStatus.NOT_FOUND)

	def set(self, chunk, result, base=0):
		"""
		Reads from the given chunk and sets the content accordingly.

		:param: chunk: the chunk to read
		:param: result: a ParseResult object.
		:param: base: the base where results' indications start at
		"""

		self.content[self.CONTENT_PARAM] = chunk[result.start + base:result.end + base]

	def analyse(self, file_obj):

		# Analyse iterates through the file object and for every chunk
		# calls the class' parse() method.
		# 
		# Depending on the result of the method, the following happens:
		#  * If the status is MAGIC_WORD_FOUND, every chunk is appended
		#    to the cached_content until FOUND happens
		#  * If the status is NOT_FOUND, the chunk is cached so that can
		#    be passed to parse() along with the next one
		#  * If the status is FOUND, set() is called, which sets the
		#    content inside the instance. Then the file object is seeked
		#    to the reported end of the block.
		#
		# If SIZE is among the StopReasons, every chunk is cached in a
		# special variable, `full_content`.

		begin = file_obj.tell()
		logger.debug("Read begins at %d" % begin)

		# If the only STOP_REASON is SIZE, then proceed in reading
		# the whole block.
		if self.STOP_REASON == StopReason.SIZE:
			# FIXME: read in chunks
			logger.debug("Avoid calling parse() as the only reason is SIZE")
			self.set(
				file_obj.read(self.size),
				ParseResult(
					status=ParseStatus.FOUND,
					end=self.size
				)
			)
			return 

		with_full_content = ((self.STOP_REASON & StopReason.SIZE) > 0)

		cached_content = b""
		cached_content_begin = 0
		full_content = b""
		while True:
			chunk = file_obj.read(self.page_size * PAGES_TO_READ)
			chunk_with_cache = cached_content + chunk
			current_size = (file_obj.tell() - begin)
			is_end = (chunk == b"")

			result = self.parse(chunk_with_cache, is_end, current_size=current_size)
			logger.debug("Got ParseResult %s" % (result,))

			if not is_end and with_full_content:
				# Append to the full cache
				full_content += chunk

			if result.status == ParseStatus.FOUND:
				# Found! Set, seek and return
				if with_full_content:
					# FIXME
					result = ParseResult(status=result.status, start=0, end=cached_content_begin+result.end)
					self.set(full_content, result, base=0)
				else:
					self.set(chunk_with_cache, result)

				boundary = begin + result.end
				logger.debug("Seeking to %d, begin is %d, result.end is %d" % (boundary, begin, result.end))
				file_obj.seek(boundary)
				self.found = True
				return
			elif result.status == ParseStatus.MAGIC_WORD_FOUND:
				# Begin caching every chunk
				cached_content = chunk_with_cache
			elif result.status == ParseStatus.NOT_FOUND:
				# Cache this page
				cached_content = chunk
				cached_content_begin = current_size - len(chunk)

			if is_end:
				# If this is the end and we are here, nothing has been
				# found.
				return

	def dump(self):
		"""
		Dumps the block.

		:returns: the block
		"""

		return self[self.CONTENT_PARAM]

	def write(self, file_obj):
		"""
		Writes the block to an already-opened file object.

		:param: file_obj: the file_obj that we should write on.
		"""

		file_obj.write(self.dump())

	def __repr__(self):
		"""
		Returns a string representation of the object.

		:returns: a string containing an helpful representation of
		the object.
		"""

		return "<%s: size %d bytes>" % (self.__class__.__name__, self.size)

class StructBlock(BaseBlock):

	STOP_REASON = StopReason.STRUCT_END

	def __init__(self, page_size=DEFAULT_PAGE_SIZE):
		"""
		Initialises the class.

		:param: page_size: the page size to use (defaults to DEFAULT_PAGE_SIZE)
		"""

		super().__init__(page_size=page_size)

		self.__params_without_padding = [
			x
			for x, y in self.PARAMS.items()
			if not y.name == "PAD"
		]

		self.__cached_format = None
		self.__cached_size = None

	@property
	def format(self):
		"""
		Returns the full format of the block.
		"""

		if not self.__cached_format:
			self.__cached_format = "".join(
				[
					obj.value
					for name, obj in self.PARAMS.items()
				]
			)

		return self.__cached_format

	@property
	def size(self):
		"""
		Returns the full size of the block.
		"""

		if not self.__cached_size:
			self.__cached_size = struct.calcsize(self.format)

		return self.__cached_size

	def __getitem__(self, item):
		"""
		Returns the item from the content dictionary, or padding bytes
		if it doesn't exist.
		"""

		if not self.content.get(item, None) is None:
			return self.content[item]

		if not item in self.PARAMS:
			raise AttributeError("%s not in PARAMS" % item)

		return b"\x00" * struct.calcsize(self.PARAMS[item].value)

	def __setitem__(self, item, value):

		self.content[item] = value

	def parse(self, chunk, is_end, current_size=None):
		"""
		Parses a chunk of bytes.

		:param: chunk: the chunk to parse
		:param: is_end: True if the file reached its end, False if not
		:returns: if successful, returns the bytes to seek to position
		the cursor just at the end of the block. In case of failure, it
		returns -1.
		"""

		if not is_end and self.STOP_REASON == StopReason.STRUCT_END:
			magic_start = chunk.find(self.MAGIC_WORD)
			if magic_start >= 0:
				# Found the magic word!
				end = magic_start + self.size
				if end < len(chunk):
					# Everything is into the boundaries of what we've read
					return ParseResult(status=ParseStatus.FOUND, start=magic_start, end=end)
				else:
					return ParseResult(status=ParseStatus.MAGIC_WORD_FOUND)

		return ParseResult(status=ParseStatus.NOT_FOUND)

	def set(self, chunk, result, base=0):
		"""
		Reads from the given chunk and sets the content accordingly.

		:param: chunk: the chunk to read
		:param: result: a ParseResult object.
		:param: base: the base where results' indications start at
		"""

		for attr_name, value in zip(
			self.__params_without_padding,
			struct.unpack(self.format, chunk[result.start + base:result.end + base])
		):
			self.content[attr_name] = value

	def dump(self):
		result = b""
		for attr_name, attr_format in self.PARAMS.items():
			if attr_format.name == "PAD":
				result += struct.pack(attr_format.value)
			else:
				result += struct.pack(attr_format.value, self[attr_name])

		return result

class DelimitedBlock(BaseBlock):

	"""
	A block with a known size.
	"""

	CONTENT_PARAM = "content"

	STOP_REASON = StopReason.SIZE

	PARAMS = {
		"content": None
	}

	def __init__(self, size, page_size=DEFAULT_PAGE_SIZE):
		"""
		Initialises the class.

		:param: size: the size of the block
		"""

		super().__init__(page_size=page_size)

		self.__cached_size = None

		if size is not None:
			# Some childs (HeaderizedBlock) pick the size via other
			# means
			self.size = size

	@property
	def size(self):

		if self.__cached_size is not None:
			return self.__cached_size

		return 0

	@size.setter
	def size(self, new_size):
		"""
		Sets the new size.
		"""

		self.__cached_size = new_size
		self.PARAMS[self.CONTENT_PARAM] = StructEnum.PAD * new_size

class Padding(BaseBlock):

	"""
	Padding.
	"""

	def get_remaining(self, file_obj):
		"""
		Returns the remaining bytes to write in order to properly pad
		the section.

		:param: file_obj: the opened file object
		:returns: the number of bytes to write
		"""

		mod = file_obj.tell() % self.page_size

		if mod > 0:
			return self.page_size - mod
		else:
			return 0

	def analyse(self, file_obj):
		"""
		Analyses the supplied file object.
		"""

		remaining = self.get_remaining(file_obj)
		
		logger.debug("Padding: skipping %d bytes" % remaining)

		file_obj.seek(
			file_obj.tell() + remaining
		)

	def dump(self):
		"""
		Useless here.
		"""

		raise Exception("Padding.dump() is unsupported. Use write() instead.")

	def write(self, file_obj):
		"""
		Writes the padding to the file.

		:param: file_obj: the file where the padding should be written
		to
		"""

		remaining = self.get_remaining(file_obj)

		logger.debug("Padding: writing %d bytes" % remaining)

		file_obj.write(b"\x00" * remaining)

class Header(StructBlock):

	"""
	The boot.img header. This is composed as follows:

	 ___________________________
	| Magic Word                |
	| Kernel Size               |
	| Kernel load address       |
	| Initramfs Size            |
	| Initramfs load address    |
	| Second image size         |
	| Second image load address |
	| Kernel tags (?) address   |
	| Page size                 |
	| null                      |
	| Product name              |
	| cmdline                   |
	| IMG ID                    |
	-----------------------------

	"""

	MAGIC_WORD = b"ANDROID!"

	PARAMS = OrderedDict(
		[
			("magic_word", StructEnum.STRING * 8),
			("kernel_size", StructEnum.UNSIGNED_INTEGER),
			("kernel_load_address", StructEnum.UNSIGNED_INTEGER),
			("initramfs_size", StructEnum.UNSIGNED_INTEGER),
			("initramfs_load_address", StructEnum.UNSIGNED_INTEGER),
			("second_image_size", StructEnum.UNSIGNED_INTEGER),
			("second_image_load_address", StructEnum.UNSIGNED_INTEGER),
			("kernel_tags_load_address", StructEnum.UNSIGNED_INTEGER),
			("page_size", StructEnum.UNSIGNED_INTEGER),
			("pad", StructEnum.PAD * 8),
			("product_name", StructEnum.STRING * 16),
			("cmdline", StructEnum.STRING * 512),
			("img_id", StructEnum.STRING * 32),
			("cmdline_extra", StructEnum.STRING * 1024)
		]
	)

	DEFAULTS = dict(
		[
			("magic_word", MAGIC_WORD),
			("kernel_size", 0),
			("initramfs_size", 0),
			("second_image_size", 0), # TODO: support second image
			("page_size", DEFAULT_PAGE_SIZE)
		]
	)

	@property
	def page_size(self):
		"""
		Hack that allows to get the embedded page_size of the header,
		if available.
		"""

		if "page_size" in self.content:
			return self.content["page_size"]

		return DEFAULT_PAGE_SIZE

	@page_size.setter
	def page_size(self, value):
		"""
		Sets the page size in the content too.
		"""

		if not hasattr(self, "content"):
			self.content = {}
			self.content.update(self.DEFAULTS)
		self.content["page_size"] = value

	def __setitem__(self, item, value):
		"""
		Override to properly handle cmdline.
		"""

		if item == "cmdline_extra":
			raise Exception("Please do not manually set cmdline_extra, use cmdline instead.")
		elif item != "cmdline":
			return super().__setitem__(item, value)

		value_length = len(value)

		if value_length > 512+1024: # cmdline + cmdline_extra
			raise Exception("Value too long")
		elif value_length <= 512:
			# Fits in cmdline
			self.content["cmdline"] = value
			self.content["cmdline_extra"] = None
		else:
			# Should use cmdline_extra
			self.content["cmdline"] = value[:512]
			self.content["cmdline_extra"] = value[512:]

class HeaderizedBlock(DelimitedBlock):

	"""
	A delimited block with support for getting sizes from an Header
	block.
	"""

	# Change this to the field of the header to lookup when
	# looking at the block size, e.g. "kernel_size".
	SIZE_FIELD = None

	def __init__(self, header, page_size=None):
		"""
		Initialises the class.

		:param: header: an Header() object
		:param: page_size: the page size to use, or None (default).
		If None is specified, the page_size is taken from the Header.
		constant.
		"""

		super().__init__(None, page_size=page_size or header.page_size)

		self.header = header

	@property
	def size(self):
		"""
		Returns the size of the block by looking at the header.

		:returns: the block size.
		"""

		return self.header[self.SIZE_FIELD]

	@size.setter
	def size(self, value):
		"""
		Sets the block size.

		:param: value: the size to set
		"""

		self.header[self.SIZE_FIELD] = value

class DeviceTree(BaseBlock):

	"""
	A Device Tree (DTB).
	"""

	MAGIC_WORD = b"\xd0\x0d\xfe\xed"

	STOP_REASON = StopReason.WORD | StopReason.END

	STOP_WORD = MAGIC_WORD

	PARAMS = {
		"content": StructEnum.PAD
	}

class Kernel(HeaderizedBlock):

	"""
	The Kernel image
	"""

	CONTENT_PARAM = "kernel"

	STOP_REASON = StopReason.WORD | StopReason.SIZE

	STOP_WORD = DeviceTree.MAGIC_WORD

	PARAMS = {
		"kernel" : None,
		"dtbs" : []
	}

	SIZE_FIELD = "kernel_size"

	def __init__(self, *args, **kwargs):
		"""
		Initialises the class.

		:param: *args: the args to pass on to the parent constructor
		:param: **kwargs: the kwargs to pass on to the parent constructor
		"""

		super().__init__(*args, **kwargs)

		self._kernel_real_size = None

	def analyse(self, file_obj):
		"""
		Analyses the object.
		"""

		# Start by getting the innerkernel
		starts_at = file_obj.tell()

		inner_kernel = InnerKernel(self.size)
		inner_kernel.analyse(file_obj)

		# Set the found kernel...
		self.content["kernel"] = inner_kernel["kernel"]
		self.content["dtbs"] = []

		# When we get here, there are two situations:
		# The first is were no DTBs were found so we are at the
		# boundary of what we can read;
		# The second is that a DTB is found so we need to keep digging
		ends_at = file_obj.tell()
		logger.debug("Kernel: ends at %d" % ends_at)

		# Update _kernel_real_size
		self._kernel_real_size = ends_at - starts_at
		logger.debug(
			"Kernel: starts at %d, real size is %d, size with DTBs is %d" % (
				starts_at,
				self._kernel_real_size,
				self.size
			)
		)

		if self.size - self._kernel_real_size > 0:
			# We should hunt for DTBs...
			# Create a BytesIO object with the rest of the kernel image.
			# We don't care about chunking as it's going to be a few KBs
			# anyways
			logger.debug("Kernel: Searching for DTBs...")
			rest = file_obj.read(self.size - self._kernel_real_size)
			with io.BytesIO(rest) as f:
				while True:
					devicetree = DeviceTree()
					devicetree.analyse(f)

					if devicetree["content"] is not None:
						logger.debug("Kernel: DTB found")
						self.content.setdefault("dtbs", []).append(devicetree)
					else:
						# End of the story
						break
		else:
			logger.debug("Kernel: Skipping searching for DTBs as this kernel doesn't have any")

	def update_size(self):
		"""
		Updates the kernel size in the Header with the correct kernel
		size + DTBs.
		"""

		if self._kernel_real_size is None:
			# analyse() hasn't been called
			return

		# Update sizes
		self.size = self._kernel_real_size + sum((len(x["content"]) for x in self.content["dtbs"]))

	def dump(self):
		"""
		Dumps kernel + DeviceTree
		"""

		return self[self.CONTENT_PARAM] + b"".join((x.dump() for x in self.content.get("dtbs", [])))

class InnerKernel(DelimitedBlock):

	"""
	The actual kernel image.
	"""

	CONTENT_PARAM = "kernel"

	STOP_REASON = StopReason.WORD | StopReason.SIZE

	STOP_WORD = DeviceTree.MAGIC_WORD

	PARAMS = {
		"kernel" : None,
	}

class Initramfs(HeaderizedBlock):

	"""
	The initramfs image
	"""

	CONTENT_PARAM = "initramfs"

	PARAMS = {
		"initramfs" : None,
	}

	SIZE_FIELD = "initramfs_size"

class BootImgContext:

	"""
	A boot.img context
	"""

	def __init__(self, image=None, page_size=DEFAULT_PAGE_SIZE):
		"""
		Initialises the class.
		"""

		self.image = image
		self.page_size = page_size

		# Image structure
		self.header = Header()
		self.kernel = Kernel(self.header)
		self.initramfs = Initramfs(self.header)
		self.padding = Padding()
		# TODO: second image support

	@property
	def blocks(self):
		"""
		Returns the blocks.
		"""
	
		for x in [
			self.header,
			self.padding,
			self.kernel,
			self.padding,
			self.initramfs,
			self.padding
		]:
			yield x

	def load(self):
		"""
		Loads the image, if specified.
		"""

		if self.image is None:
			# Nothing to load
			return

		with open(self.image, "rb") as f:
			for block in self.blocks:
				logger.debug("Loading block %s" % block)
				# This allows the pagesize to be adjusted after reading
				# the header
				block.page_size = self.header.page_size

				block.analyse(f)

	def dump_to(self, to, what=[DumpAction.EVERYTHING]):
		"""
		Dumps the components of the image in the specified directory.

		:param: to: the directory where to store the files
		:param: what: a list of DumpActions. Defaults to [DumpAction.EVERYTHING]
		"""

		# TODO:

		action_map = {
			DumpAction.HEADER : self.header,
			DumpAction.KERNEL : self.kernel,
			DumpAction.INITRAMFS : self.initramfs,
			DumpAction.DTBS : lambda x: None,
		}

		if not os.path.exists(to):
			os.makedirs(to)

		if DumpAction.EVERYTHING in what:
			what = [x for x in DumpAction.__members__.values() if x not in (DumpAction.EVERYTHING, DumpAction.NOTHING)]

		for action in what:
			target_file = os.path.join(to, action.value)

			if action == DumpAction.DTBS:
				# DTBs are special
				counter = 1
				for dtb in self.kernel["dtbs"]:
					# TODO: get name
					with open(os.path.join(to, "dtb_%d" % counter), "wb") as f:
						dtb.write(f)
					counter += 1
			else:
				with open(target_file, "wb") as f:
					action_map[action].write(f)

	def update_img_id(self):
		"""
		Updates the img_id in the header.
		"""

		hasher = hashlib.sha1()
		for block in [self.kernel, self.initramfs]:
			hasher.update(block.dump())
			hasher.update(struct.pack(StructEnum.UNSIGNED_INTEGER.value, block.size))
		hasher.update(struct.pack(StructEnum.UNSIGNED_INTEGER.value, 0)) # FIXME: second image size

		self.header["img_id"] = hasher.digest()

	def dump(self, file_obj):
		"""
		Dumps the image to the specified file object.

		:param: file_obj: the file_obj to dump the image on.
		"""

		for block in self.blocks:
			logger.debug("Context: dumping %s to %s" % (block, file_obj.name))
			block.write(file_obj)

	def __enter__(self):
		"""
		Enters in the context.
		"""

		return self

	def __exit__(self, *args, **kwargs):
		pass

parser = argparse.ArgumentParser()
parser.add_argument(
	"--debug",
	help="enables debug logging",
	action="store_true",
	default=False
)
parser.add_argument(
	"--input", "-i",
	help="the input image path",
	type=str
)
parser.add_argument(
	"--output", "-o",
	help="the path where to store the output",
	type=str
)
parser.add_argument(
	"--dump", "-u",
	help="what to dump. Might be specified multiple times. Defaults to nothing",
	type=DumpAction,
	nargs="+",
	choices=list(DumpAction), #[x.value for x in DumpAction.__members__.values()],
)
parser.add_argument(
	"--dump-to", "-t",
	help="where to dump the specified images. Defaults to the current directory",
	type=str,
	default="."
)
parser.add_argument(
	"--print-header", "-e",
	help="prints the header",
	action="store_true",
	default=False
)
parser.add_argument(
	"--cmdline", "-c",
	help="use the supplied cmdline",
	type=lambda x: bytes(x, "ascii")
)
parser.add_argument(
	"--remove-original-dtbs", "-r",
	help="removes the original dtbs found in the kernel",
	action="store_true"
)
parser.add_argument(
	"--dtb",
	help="appends the specified dtb to the kernel. It might be used multiple times",
	nargs="+",
	type=str
)
parser.add_argument(
	"--kernel",
	help="uses the specified kernel image",
	type=str
)
parser.add_argument(
	"--initramfs",
	help="uses the specified initramfs image",
	type=str
)
parser.add_argument(
	"--base",
	help="the base address to use",
	type=lambda x: int(x, 0),
	default=DEFAULT_BASE
)
parser.add_argument(
	"--kernel-offset",
	help="the kernel offset",
	type=lambda x: int(x, 0),
)
parser.add_argument(
	"--initramfs-offset",
	help="the initramfs offset",
	type=lambda x: int(x, 0),
	#default=0x01000000
)
parser.add_argument(
	"--second-offset",
	help="the second bootloader offset",
	type=lambda x: int(x, 0),
)
parser.add_argument(
	"--tags-offset",
	help="the tags offset",
	type=lambda x: int(x, 0),
)
parser.add_argument(
	"--pagesize", "-p",
	help="the page size to use",
	choices=[2**i for i in range(11,15)],
	default=None
)

if __name__ == "__main__":

	args = parser.parse_args()

	logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

	with BootImgContext(image=args.input, page_size=args.pagesize) as context:
		context.load()

		if args.cmdline:
			context.header["cmdline"] = args.cmdline

		if args.pagesize is not None:
			context.header["page_size"] = args.pagesize

		if args.kernel_offset is not None:
			context.header["kernel_load_address"] = args.base + args.kernel_offset
		elif not "kernel_load_address" in context.header:
			# Set default
			context.header["kernel_load_address"] = args.base + DEFAULT_KERNEL_OFFSET

		if args.initramfs_offset is not None:
			context.header["initramfs_load_address"] = args.base + args.initramfs_offset
		elif not "initramfs_load_address" in context.header:
			# Set default
			context.header["initramfs_load_address"] = args.base + DEFAULT_INITRAMFS_OFFSET

		if args.tags_offset is not None:
			context.header["kernel_tags_load_address"] = args.base + args.tags_offset
		elif not "kernel_tags_load_address" in context.header:
			# Set default
			context.header["kernel_tags_load_address"] = args.base + DEFAULT_TAGS_OFFSET

		if args.second_offset is not None:
			context.header["second_image_load_address"] = args.base + args.second_offset
		elif not "second_image_load_address" in context.header:
			# Set default
			context.header["second_image_load_address"] = args.base + DEFAULT_SECOND_IMAGE_OFFSET

		if args.kernel is not None:
			with open(args.kernel, "rb") as f:
				context.header["kernel_size"] = os.path.getsize(args.kernel)
				new_kernel = Kernel(context.header)
				new_kernel.analyse(f)
				context.kernel = new_kernel

		if args.initramfs is not None:
			with open(args.initramfs, "rb") as f:
				context.header["initramfs_size"] = os.path.getsize(args.initramfs)
				new_initramfs = Initramfs(context.header)
				new_initramfs.analyse(f)
				context.initramfs = new_initramfs

		if args.remove_original_dtbs:
			context.kernel.content["dtbs"] = []

		if args.dtb is not None:
			for dtb in args.dtb:
				dt_obj = DeviceTree()
				with open(dtb, "rb") as f:
					dt_obj.content["content"] = f.read()
				context.kernel.content["dtbs"].append(dt_obj)

		# Update sizes
		context.kernel.update_size()

		if False in (context.kernel.size > 0, context.initramfs.size > 0):
			raise Exception("The bootimage requires at least a kernel and an initramfs")

		# Update img_id
		context.update_img_id()

		# Print header
		if args.print_header:
			for component in context.header.PARAMS:
				print("%s: %s" % (component, context.header[component]))

		# Dump
		if args.dump and not DumpAction.NOTHING in args.dump:
			context.dump_to(args.dump_to, what=args.dump)

		# Output
		if args.output:
			with open(args.output, "wb") as output:
				context.dump(output)

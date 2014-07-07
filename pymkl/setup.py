#!/usr/bin/python
from distutils.core import setup, Extension
import sys

setup(name="adder",
	version="0.1",
	ext_modules = [Extension("adder",["adder.c"])])



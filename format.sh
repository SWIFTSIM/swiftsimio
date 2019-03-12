#!/bin/bash

# Formats the code.

black *.py
black tests/*.py
black swiftsimio/*.py
black swiftsimio/*/*.py
black swiftsimio/*/*/*.py

#!/bin/bash

# 1. Set ACLs for the directory itself
setfacl --set=u::rwx,g::rwx,g:hhb:rwx,o::---,m::rwx .

# 2. Set default ACLs for future directories and files
setfacl -m d:u::rwx,d:g::rwx,d:g:hhb:rwx,d:o::---,d:m::rwx .

# 3. For existing directories recursively: rwx for user & group, nothing for others
find . -type d -exec setfacl --set=u::rwx,g::rwx,g:hhb:rwx,o::---,m::rwx {} \;

# 4. For existing files recursively: rw for user & group, nothing for others
find . -type f -exec setfacl --set=u::rw-,g::rw-,g:hhb:rw-,o::---,m::rw- {} \;
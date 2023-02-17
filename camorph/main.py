import sys

import camorph.camorph as camorph


flag = int(sys.argv[1])
read_path = sys.argv[2]
output_path = sys.argv[3]
num_of_view = int(sys.argv[4])
pose_path = sys.argv[5]


if int(flag) == 0:
	cams = camorph.read_cameras('mpeg_omaf', read_path)
	for V in range(0, num_of_view):
		cams[V].source_image = "null"
	camorph.write_cameras('nerf', output_path, cams)
else:
	cams = camorph.read_cameras('mpeg_omaf', read_path, pose_path, posetrace="posetrace")
	camorph.write_cameras("nerf", output_path, cams)


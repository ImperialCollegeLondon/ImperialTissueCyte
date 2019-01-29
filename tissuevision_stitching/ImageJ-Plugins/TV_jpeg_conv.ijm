print("\\Clear");
print("Running image stitch macro...");

var maindir = "";
channels= newArray("1","2","3");

Dialog.create("Relevant Variables");

Dialog.addNumber("Starting section in dataset:",1);
Dialog.addNumber("Ending section in dataset:",1);
Dialog.addNumber("Number of X direction mosaic steps:",1);
Dialog.addNumber("Number of Y direction mosaic steps:",1);
Dialog.addNumber("Percent overlap between steps:",10);
Dialog.addNumber("Number of Z layers per section:",1);
Dialog.addChoice("Select the channel you want stitched",channels);
Dialog.show();
slicesstrt = Dialog.getNumber();
slicesend = Dialog.getNumber();
xsteps = Dialog.getNumber();
ysteps = Dialog.getNumber();
overlap = Dialog.getNumber();
zlayers = Dialog.getNumber();
chan = Dialog.getChoice();

maindir = getDirectory("Choose Your Image-Containing folder");
print("Image directory: "+maindir);


if (zlayers == 1){
slices = slicesend-slicesstrt+1;
} else {
slices = (slicesend-slicesstrt+1)*zlayers;
	
}


if (chan == 1){
print("Starting stitching...");

imgdir = maindir+"Ch1_Tile_Sections_"+slicesstrt+"_to_"+slicesend;
stitchdir = maindir+"Ch1_Stitched_Sections_"+slicesstrt+"_to_"+slicesend;
commandstrng1 = "grid_size_x="+xsteps+" grid_size_y="+ysteps+" grid_size_z="+slices+" overlap="+overlap+" input="+imgdir+" file_names=Tile_Z{zzz}_Y{yyy}_X{xxx}.tif rgb_order=rgb output_file_name=TileConfiguration_{zzz}.txt output="+stitchdir+" start_x=1 start_y=1 start_z=1 start_i=1 channels_for_registration=[Red, Green and Blue] fusion_method=[Linear Blending] fusion_alpha=1.50 regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50";

run("Collect Garbage");
print("All images stitched.");

dirlist = getFileList(stitchdir);
nofiles = dirlist.length;

for (i=1; i<=nofiles; i++){
if(i <= 9){
tokennmbr = "00"+i;
} else {
	if(i<=99){
		tokennmbr = "0"+i;
	} else{
		tokennmbr = i;
	}
}
	openstrng = stitchdir+"/Stitched Image_"+tokennmbr+".tif";
open(openstrng);

tiffsavestrng = stitchdir+"/Stitched Image_ch1_"+tokennmbr+".tif";
saveAs("Tiff", tiffsavestrng);
wait(1000);

run("Size...", "width=1000 height=1528 constrain average interpolation=Bilinear");
wait(1000);	
run("Enhance Contrast", "saturated=0.4");
wait(1000);
	savestrng = stitchdir+"/JPEG Stitched Image_ch1_"+tokennmbr+".jpg";
	print("Saving JPEG image: "+savestrng);
	saveAs("Jpeg", savestrng);
wait(1000);
run("Select All");
run("Clear", "slice");

clearsavestrng = stitchdir+"/Stitched Image_"+tokennmbr+".tif";
saveAs("Tiff", clearsavestrng);
wait(1000);
close();
run("Collect Garbage");
}
print("Operation completed.");

} else {

	if(chan == 2){
		

imgdir2 = maindir+"Ch2_Tile_Sections_"+slicesstrt+"_to_"+slicesend;
stitchdir2 = maindir+"Ch2_Stitched_Sections_"+slicesstrt+"_to_"+slicesend;
commandstrng2 = "grid_size_x="+xsteps+" grid_size_y="+ysteps+" grid_size_z="+slices+" overlap="+overlap+" input="+imgdir2+" file_names=Tile_Z{zzz}_Y{yyy}_X{xxx}.tif rgb_order=rgb output_file_name=TileConfiguration_{zzz}.txt output="+stitchdir2+" start_x=1 start_y=1 start_z=1 start_i=1 channels_for_registration=[Red, Green and Blue] fusion_method=[Linear Blending] fusion_alpha=1.50 regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50";

run("Collect Garbage");
print("All images stitched.");
dirlist = getFileList(stitchdir2);
nofiles = dirlist.length;

for (i=1; i<=nofiles; i++){
if(i <= 9){
tokennmbr = "00"+i;
} else {
	if(i<=99){
		tokennmbr = "0"+i;
	} else{
		tokennmbr = i;
	}
}
	openstrng = stitchdir2+"/Stitched Image_"+tokennmbr+".tif";
open(openstrng);

tiffsavestrng = stitchdir2+"/Stitched Image_ch2_"+tokennmbr+".tif";
saveAs("Tiff", tiffsavestrng);
wait(1000);

run("Size...", "width=1000 height=1528 constrain average interpolation=Bilinear");
wait(1000);	
run("Enhance Contrast", "saturated=0.4");
wait(1000);
	savestrng = stitchdir2+"/JPEG Stitched Image_ch2_"+tokennmbr+".jpg";
	print("Saving JPEG image: "+savestrng);
	saveAs("Jpeg", savestrng);
wait(1000);
run("Select All");
run("Clear", "slice");

clearsavestrng = stitchdir2+"/Stitched Image_"+tokennmbr+".tif";
saveAs("Tiff", clearsavestrng);
wait(1000);
close();
run("Collect Garbage");
}

print("Operation completed.");



	} else {



imgdir3 = maindir+"Ch3_Tile_Sections_"+slicesstrt+"_to_"+slicesend;
stitchdir3 = maindir+"Ch3_Stitched_Sections_"+slicesstrt+"_to_"+slicesend;
commandstrng3 = "grid_size_x="+xsteps+" grid_size_y="+ysteps+" grid_size_z="+slices+" overlap="+overlap+" input="+imgdir3+" file_names=Tile_Z{zzz}_Y{yyy}_X{xxx}.tif rgb_order=rgb output_file_name=TileConfiguration_{zzz}.txt output="+stitchdir3+" start_x=1 start_y=1 start_z=1 start_i=1 channels_for_registration=[Red, Green and Blue] fusion_method=[Linear Blending] fusion_alpha=1.50 regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50";

run("Collect Garbage");
print("All images stitched.");
dirlist = getFileList(stitchdir3);
nofiles = dirlist.length;

for (i=1; i<=nofiles; i++){
if(i <= 9){
tokennmbr = "00"+i;
} else {
	if(i<=99){
		tokennmbr = "0"+i;
	} else{
		tokennmbr = i;
	}
}
	openstrng = stitchdir3+"/Stitched Image_"+tokennmbr+".tif";
open(openstrng);

tiffsavestrng = stitchdir3+"/Stitched Image_ch3_"+tokennmbr+".tif";
saveAs("Tiff", tiffsavestrng);
wait(1000);

run("Size...", "width=1000 height=1528 constrain average interpolation=Bilinear");
wait(1000);	
run("Enhance Contrast", "saturated=0.4");
wait(1000);
	savestrng = stitchdir3+"/JPEG Stitched Image_ch3_"+tokennmbr+".jpg";
	print("Saving JPEG image: "+savestrng);
	saveAs("Jpeg", savestrng);
wait(1000);
run("Select All");
run("Clear", "slice");

clearsavestrng = stitchdir3+"/Stitched Image_"+tokennmbr+".tif";
saveAs("Tiff", clearsavestrng);
wait(1000);
close();
run("Collect Garbage");
}

print("Operation completed.");


	}
}

run("Collect Garbage");
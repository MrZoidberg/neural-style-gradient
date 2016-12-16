#!/usr/bin/env bash

while getopts "s:c:o:" opt; do
case $opt in
s) STYLE="$OPTARG"
;;
c) SOURCE="$OPTARG"
;;
o) OUTROOT="$OPTARG"
;;
*) echo "No reasonable options found!";;
esac
done

#user paths to neural style folder scripts
userpath="$HOME/code/zoid-neural-style/"

#This steps prevents the need to change directory before running script when you open bash. You must change the directory below to the same as above
cd $HOME/code/zoid-neural-style/

############################
#style & content source
stylesource="${STYLE}"
contentsource="${SOURCE}"

echo stylesource = "$stylesource"
echo contentsource = "$contentsource"

############################
#image size, optimizer, save/print frequency & quality adam or lbfgs, adam is faster, lbfgs is higher quality
printiter="50"
saveiter="0"
numiter="300"
imagesize="300"
optimize="adam"

############################
#learningrate and normalize_gradients
learningrate="15"
maxlearningrate=$learningrate
learningratestep="5"
learningrate=learningrate-learningratestep

normalize_gradients=0

############################
#style and content weighting
#These are the starting points for your style gradient!
styleweight="100"
styleweightstep="100"
contentweight="10"
contentweightstep="1"
tvweight=".0001"
stylescale="1.8"
stylescalestep=".3"

############################
#This will set the name of the directory created or used for this project
project="${OUTROOT}/_gradient"

echo project="${project}"
############################
#Creates Directories for this project
contentsourcefile=${contentsource##*/}
stylesourcefile=${stylesource##*/}
mkdir -p $project/${contentsourcefile%.*}/${stylesourcefile%.*}
echo mkdir -p $project/${contentsourcefile%.*}/${stylesourcefile%.*}

############################
#location of scripts and models
neuralstlefile="neural_style.lua"
modelfile="models/VGG_ILSVRC_19_layers.caffemodel"
protofile="models/VGG_ILSVRC_19_layers_deploy.prototxt"
contentlayers="relu4_2"
stylelayers="relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"


# noise fractal seed
seedit="123"

# backend opencl or clnn cudnn, nn or cpu - nn works best on my machine you might need a different backend
backend="cudnn"
g_pu="0"

#initialization method (image or random)
initialize="image"

#image name root will begin with a string below
imagename="VGG19DEF"

#shortnames for file naming .jpg files
#styleweight
sw="sw_"
#$contentweight
cw="cw_"
#style scale
sc="sc_"
#tv weight
tw="tw_"
#normalaize
norm="norm_"
#learning rate
lr="lr_"
#seperators
sep="_"




#End Variable edits, Command loop(s) below

#This loop will create 4 sets of the loop below, changing only the stylscale and TV Weight. Change root degrees of style gradient by adjusting the values below.

for scc in `seq 1 4`;
do
stylescale=$(echo "scale=2;$stylescale - $stylescalestep" | bc)
tvweight=$(echo "scale=5;$tvweight +.0001" | bc)

#This step creates seven style gradient steps for each of the loops above, adjusting only styleweight and contentweight. Change the gradient steps by adjusting the values below.

for q in `seq 1 7`;
do
styleweight=$((styleweight+styleweightstep))
contentweight=$((contentweight-contentweightstep))
#This loop runs turning off and on normalize_gradients

for n in `seq 1 4`;
do
	if [[ $normalize_gradients -eq "0" && $learningrate -ge $maxlearningrate]]; then
		$normalize_gradients=1
	fi

	if [[ $normalize_gradients -e "1"]]; then
		$learningrate=$learningrate-$learningratestep
  fi

	if [[ $normalize_gradients -e "1" && l$learningrate -lt $(($maxlearningrate - $learningratestep*3))]]; then
		$normalize_gradients=0
	fi

	echo normalize_gradients = $normalize_gradients
	echo learningrate = $learningrate

#This loop runs the neural style with the adjusted parameters from the loops above, The total number of images will be the length of first loop multiplied by length of second loop.

for i in `seq 1 1`;

do
    # scoping loop (i expands here now)

		thisrun = $sw$styleweight$sep$cw$contentweight$sep$sc$stylescale$sep$tw$tvweight$sep$norm$normalize_gradients$sep$lr$learningrate

		CMDone="th $userpath$neuralstlefile
				-style_image $stylesource
				-content_image $contentsource
				-output_image $project/${contentsourcefile%.*}/${stylesourcefile%.*}/$imagename$sep$thisrun.jpg
				-model_file $userpath$modelfile
				-proto_file $userpath$protofile
				-content_layers $contentlayers
				-style_layers $stylelayers
				-gpu $g_pu
				-backend $backend
				-seed $seedit
				-content_weight $contentweight
				-print_iter $printiter
				-style_weight $styleweight
				-optimizer $optimize
				-image_size $imagesize
				-num_iterations $numiter
				-style_scale $stylescale
				-tv_weight $tvweight
				-save_iter $saveiter
				-init $initialize
				-normalize_gradients $normalize_gradients
				-learning_rate $learning_rate"

    # display it for the joy of looking at it
		echo $CMDone

    # run it, because that's what the script is for
		$CMDone
done
done
done
done

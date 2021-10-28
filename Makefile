include Makefile.inc

build: $(BIN)/$(HL_TARGET)/main

#IMAGE_NAME = "Rodrigo Valla - CC BY-NC 2.0.jpg"
#IMAGE_NAME = "a4608-Duggan_080413_6147_N1.5.JPG"
#IMAGE_NAME = "a0716-MB_20030906_030_P1.5.JPG"
#IMAGE_NAME = "a0964-20061224_140610__E6B7217_N1.5.JPG"
#IMAGE_NAME = "a1359-NKIM_MG_6126_N1.5.JPG"
#IMAGE_NAME = "a1475-dgw_146_P1.JPG"
#IMAGE_NAME = "a1956-JI2E3597_N1.5.JPG"
#IMAGE_NAME = "a2039-dvf_088_N1.5.JPG"
#IMAGE_NAME = "a2192-IMG_2045_P1.5.JPG"
#IMAGE_NAME = "a2490-WP_CRW_3690_N1.5.JPG"
#IMAGE_NAME = "a2682-DSC_0085_P1.5.JPG"
#IMAGE_NAME = "a3354-MB_070908_069_P1.5.JPG"
#IMAGE_NAME = "a4747-09-05-19-at-19h04m32s-_MG_9562_P1.5.JPG"
#IMAGE_NAME = "a4874-20090320_at_19h15m51__MG_0126_N1.5.JPG"
#IMAGE_NAME = "a4875-DSC_0382_N1.5.JPG"
#IMAGE_NAME = "Floris van Lint-CC BY-NC 2.0.jpg"
#IMAGE_NAME = "frostnip907 Flickr CC BY-NC-SA 2.0.jpg"
#IMAGE_NAME = "julochka Flickr CC BY-NC 2.0.jpg"
#IMAGE_NAME = "Justin Chiaratti-CC BY-NC-SA 2.0.jpg"
#IMAGE_NAME = "Probably Okay! (photo4) CC BY-SA 2.0.jpg"
IMAGE_NAME = "y Gabriele Flickr CC BY-NC-SA 2.0.jpg"

$(BIN)/%/main: main.cpp src/util.cpp src/ml_tools.cpp src/model.cpp src/load_model.cpp src/fit_and_slice.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -I./src -Wall $^ -o $@ $(LDFLAGS) $(LIBHALIDE_LDFLAGS) $(IMAGE_IO_FLAGS)

test: $(BIN)/$(HL_TARGET)/main
	$^ ml_variables example_images $(IMAGE_NAME)
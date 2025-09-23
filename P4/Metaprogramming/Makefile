PNG_FILES := $(wildcard *.png)
JPG_FILES := $(PNG_FILES:.png=.jpg)

all: $(JPG_FILES)

%.jpg: %.png
	convert $< $@


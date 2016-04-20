package main

//#cgo pkg-config: opencv
//#include <cv.h>
//#include <highgui.h>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

import (
	"C"
	"fmt"
	// "unsafe"
	// "io"
	// "io/ioutil"
	// "os"
	// "path"
	// "runtime"
)

var perfDebug bool = true
var fileWrite bool = true
var displayProcessed bool = true

func main() {
	uri := "/home/solomon/frc/the-deal/distance.jpg"
	text := C.CString("Hello, World!")
	defer C.free(unsafe.Pointer(text))
	img := unsafe.Pointer(C.cvCreateImage(C.cvSize(640, 480)))
}

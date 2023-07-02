<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <h1>Image Tagger</h1>
        <p class="py-1">
          This is a simple image tagging application. It allows you to upload
          images and tag them with keywords.
        </p>
        <p class="py-1">
          The model prediction is done on the client browser using ONNX runtime.
        </p>
        <p class="py-1">
          Your images are not sent to any server and your privacy is respected.
        </p>
      </v-col>
    </v-row>
    <v-row>
      <v-col cols="12">
        <v-file-input id="file-in" label="File input" class="max-30-width" @change="onFileChange"></v-file-input>
        <br>
        <div class="py-10">
          <v-progress-circular v-if="isRunning" indeterminate :size="37" :width="5"></v-progress-circular>
          <v-chip v-for="tag in tags" :key="tag" prepend-icon="mdi-tag" size="large" class="mx-1">
            {{ tag }}
          </v-chip>
        </div>
      </v-col>
      <v-col cols="6">
        <img src="/nuculabs-logo.png" id="input-image" class="fixed-image" alt="Input image"/>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import {defineComponent} from "vue";
import {InferenceSession, Tensor} from "onnxruntime-web";
import classes from "@/model/classes";

const WIDTH = 224;
const DIMS = [1, 3, WIDTH, WIDTH];
const MAX_LENGTH = DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3];
const MAX_SIGNED_VALUE = 255.0;
const MODEL_PATH = "/image-tagger-vue/resnet34_10_epochs.onnx";

export default defineComponent({
  name: "HomeView",
  data() {
    return {
      session: null,
      isRunning: false,
      tags: [],
    }
  },
  methods: {
    // Fires when the user selects an image file.
    onFileChange(event) {
      this.isRunning = true;

      let target = event.target || window.event.src;
      let files = target.files;

      if (FileReader && files && files.length) {
        let fileReader = new FileReader();
        fileReader.onload = () => this.onLoadImage(fileReader);
        fileReader.readAsDataURL(files[0]);
      }
    },
    // Fires when the image is loaded into image element.
    // Updates the image element with user file.
    onLoadImage(fileReader) {
      let img = document.getElementById("input-image");
      img.onload = () => this.handleImage(img, WIDTH);
      img.src = fileReader.result;
    },
    // Runs the model on the image.
    async handleImage(img, targetWidth) {
      const resizedImageData = await this.processImage(img, targetWidth);
      const inputTensor = await this.imageDataToTensor(resizedImageData, DIMS);
      await this.runModel(inputTensor);
    },
    // Resizes the image to a square of size specified by width. 224 in this case.
    async processImage(img, width) {
      const canvas = document.createElement("canvas");
      let ctx = canvas.getContext("2d");

      canvas.width = width;
      canvas.height = canvas.width * (img.height / img.width);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      return ctx.getImageData(0, 0, width, width).data;
    },
    // Transforms the image data into a tensor.
    async imageDataToTensor(data, dims) {
      // 1. filter out alpha
      // 2. transpose from [224, 224, 3] -> [3, 224, 224]
      const [R, G, B] = [[], [], []];
      for (let i = 0; i < data.length; i += 4) {
        R.push(data[i]);
        G.push(data[i + 1]);
        B.push(data[i + 2]);
        // here we skip data[i + 3] because it's the alpha channel
      }
      const transposedData = R.concat(G).concat(B);

      // convert to float32
      let l = transposedData.length; // length, we need this for the loop
      const float32Data = new Float32Array(MAX_LENGTH); // create the Float32Array for output
      for (let i = 0; i < l; i++) {
        float32Data[i] = transposedData[i] / MAX_SIGNED_VALUE; // convert to float
      }

      // return ort.Tensor
      return new Tensor("float32", float32Data, dims);
    },
    async runModel(inputTensor) {
      try {
        const session = await InferenceSession.create(MODEL_PATH);
        const feeds = {"input.1": inputTensor};

        // feed inputs and run
        const results = await session.run(feeds);

        let predictedClasses = [];
        for (let i = 0; i < results["368"].data.length; i++) {
          let value = results["368"].data[i];
          if (value > -0.5) {
            predictedClasses.push(classes[i]);
          }
        }
        this.tags = predictedClasses;
      } catch (e) {
        console.error(e);
      } finally {
        this.isRunning = false;
      }
    }
  },
})
</script>

<style>
.max-30-width {
  max-width: 30vw;
}

.fixed-image {
  height: 224px;
  width: 224px;
}
</style>

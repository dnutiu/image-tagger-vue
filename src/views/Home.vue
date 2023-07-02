<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <h1>Image Tagger</h1>
        <p class="py-1">
          This is a simple image tagging application. It allows you to upload
          images and tag them keywords.
        </p>
        <p class="py-1">
          The application is built using Vue.js and Vuetify.
          The model prediction is done on the client side using ONNX runtime.
        </p>
        <p class="py-1">
          Your images are not sent to any server and your privacy is respected.
        </p>
      </v-col>
    </v-row>
    <v-row>
      <v-col cols="12">
        <input type="file" id="file-in" name="file-in">
        <br>
        <div class="py-10">
          <v-chip v-for="tag in tags" :key="tag" prepend-icon="mdi-tag" size="large" class="mx-1">
            {{ tag }}
          </v-chip>
        </div>
      </v-col>
      <v-col cols="6">
        <img src="" id="input-image" class="fixed-image" alt="Input image" />
        <img src="" class="hidden" id="canvas-image" alt="Input image" style="display: none" />
      </v-col>
      <v-col cols="6">
        <img src="" id="scaled-image" class="scaled-image" style="display: none"  alt="scaled image"/>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
import {defineComponent} from "vue";
import {InferenceSession, Tensor} from "onnxruntime-web";
import classes from "@/model/classes";
export default defineComponent({
  name: "HomeView",
  data() {
    return {
      session: null,
      isRunning: false,
      tags: [],
    }
  },
  methods: {},
  async mounted() {

    const WIDTH = 224;
    const DIMS = [1, 3, WIDTH, WIDTH];
    const MAX_LENGTH = DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3];
    const MAX_SIGNED_VALUE = 255.0;

    let predictedClass;
    let isRunning = false;

    function onLoadImage(fileReader) {
      var img = document.getElementById("input-image");
      img.onload = () => handleImage(img, WIDTH);
      img.src = fileReader.result;
    }

    function handleImage(img, targetWidth) {
      ctx.drawImage(img, 0, 0);
      const resizedImageData = processImage(img, targetWidth);
      const inputTensor = imageDataToTensor(resizedImageData, DIMS);
      run(inputTensor);
    }

    function processImage(img, width) {
      const canvas = document.createElement("canvas"),
        ctx = canvas.getContext("2d");

      canvas.width = width;
      canvas.height = canvas.width * (img.height / img.width);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      document.getElementById("canvas-image").src = canvas.toDataURL();
      return ctx.getImageData(0, 0, width, width).data;
    }

    function imageDataToTensor(data, dims) {
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
      let i,
        l = transposedData.length; // length, we need this for the loop
      const float32Data = new Float32Array(MAX_LENGTH); // create the Float32Array for output
      for (i = 0; i < l; i++) {
        float32Data[i] = transposedData[i] / MAX_SIGNED_VALUE; // convert to float
      }

      // return ort.Tensor
      const inputTensor = new Tensor("float32", float32Data, dims);
      return inputTensor;
    }

    let self = this;
    async function run(inputTensor) {
      try {
        const session = await InferenceSession.create("resnet34_10_epochs.onnx");
        const feeds = { "input.1": inputTensor };

        // feed inputs and run
        const results = await session.run(feeds);

        let predictedClasses = [];
        for (let i = 0; i < results["368"].data.length; i++) {
          let value = results["368"].data[i];
          if (value > -0.5) {
            predictedClasses.push(classes[i]);
          }
        }

        predictedClass = `${predictedClasses}`;
        self.tags = predictedClasses;
        isRunning = false;
      } catch (e) {
        console.error(e);
        isRunning = false;
      }
    }

    const canvas = document.createElement("canvas")
    let ctx = canvas.getContext("2d");

    document.getElementById("file-in").onchange = function (evt) {
      let target = evt.target || window.event.src,
        files = target.files;

      if (FileReader && files && files.length) {
        isRunning = true;
        let fileReader = new FileReader();
        fileReader.onload = () => onLoadImage(fileReader);
        fileReader.readAsDataURL(files[0]);
      }
    };
  },
})
</script>

<style>
.fixed-image {
  height: 224px;
  width: 224px;
}
</style>

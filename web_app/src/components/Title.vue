<template>
    <div class="image-upload-container">
        <div class="title">
            <label for="file-input" class="image-preview" v-if="selectedFile">
                <img :src="imagePreviewUrl" alt="Selected Image Preview" />
            </label>
            <input id="file-input" type="file" @change="handleFileChange" style="display: none" />
            <label for="file-input" class="upload-button">
                <span v-if="!selectedFile">+</span>
                <span v-else>Edit</span>
            </label>
        </div>
        <div class="bank">
            <button @click="uploadImage" v-if="selectedFile">Upload</button>
        </div>
        <div class="upload-section">

<!--            <button @click="uploadImage" v-if="selectedFile">Upload</button>-->
        </div>

    </div>

</template>

<script>
import axios from 'axios';

export default {
    name: 'PageTitle',
    props: {
        msg: String,
    },
    data() {
        return {
            selectedFile: null,
            imagePreviewUrl: '',
        };
    },
    methods: {
        handleFileChange(event) {
            this.selectedFile = event.target.files[0];
            this.previewImage();
        },
        uploadImage() {
            const formData = new FormData();
            formData.append('image', this.selectedFile);
            axios
                .post('http://upload', formData)
                .then((response) => {
                    console.log('Image uploaded successfully:', response.data);
                })
                .catch((error) => {
                    console.error('Error uploading image:', error);
                });
        },
        previewImage() {
            const reader = new FileReader();

            reader.onload = (e) => {
                this.imagePreviewUrl = e.target.result;
            };

            reader.readAsDataURL(this.selectedFile);
        },
    },
};
</script>

<style scoped>
.image-upload-container {
    display: flex;
    height: 100vh;
   margin-top: 100px;
}

.title {
    display: flex;
    flex: 5; /* Occupy one-third of the width */
    z-index: 5;
    background-color: #42b983;
    color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    align-items: center;
    margin-left: 20px;
}
.bank{
    display: flex;
    flex: 1; /* Occupy one-third of the width */
    z-index: 5;
    padding: 20px;
    border-radius: 8px;
    align-items: center;
}
.upload-section {
    flex: 5; /* Occupy two-thirds of the width */
    flex-direction: column;
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
    margin-right: 20px;
}

.image-preview {
    display: inline-block;
    position: relative;
    /*width: 200px;*/
    /*height: 200px;*/
    border: 2px dashed #ccc;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    margin-bottom: 10px; /* Add margin between image preview and button */
}

.image-preview img {
    width: 550px;
    height: 550px;

}

.upload-button {
    display: flex;
    background-color: white;
    /*color: white;*/
    font-size: 24px;
    width: 550px;
    height: 550px;
    line-height: 50px;
    border-radius: 8px;
    cursor: pointer;
    margin-top: 10px;
    text-align: center;
}

.upload-button span {
    display: block;
    text-align: center;
}

button {
    background-color: #42b983;
    color: white;
    padding: 5px 30px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    align-self: center;
    display: flex;
}
</style>
<template>

    <!--网站首页-->
    <div class="container">
        <button class="title">Image Restoration</button>
        <p class="description" style="white-space: pre-wrap">
            {{'Upload an image contaminated by noise and return the repair result of the image.'}}
        </p>
    </div>

    <div class="common-layout">
        <el-container class="el-container">

            <el-header class="el-header">图像修复</el-header>

            <el-container class="container">
                <el-main class="el-main">

                    <el-row class="el-row" :gutter="10">
                        <el-col class="el-col" >
                            <div class="block">
                                <div class="grid-content bg-purple">

                                    <input id="file-input" type="file" @change="handleFileChange" style="display: none" accept="image/*" />
                                    <label for="file-input" class="image-preview">
                                        <img :src="imagePreviewUrl" class="image-preview" />
                                    </label>
                                </div>
                                <button class="btn" @click="uploadImage">上传需修复的图片</button>
                            </div>

                        </el-col>

                        <el-col >
                            <div class="block">
                                <div class="grid-content bg-purple">
                                    <label for="file-output" class="image-preview" >
                                        <img :src="repairedImage" class="image-preview" />
                                    </label>
                                </div>
                                <button class="btn">下载修复后的图片</button>
                            </div>
                        </el-col>
                    </el-row>

                </el-main>
            </el-container>



            <el-footer class="el-footer">第十二小组</el-footer>

        </el-container>
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
            repairedImage:null,
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
                    this.repairedImage=response.data.url;
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
<style >
.container{
    height: 100vh;
    background-color: #633fff;
    display: flex;
    flex-direction: column;
    justify-content: center; /* 水平居中 */
    align-items: center; /* 垂直居中 */
}
.title {
    font-size: 7rem;
    padding: 3rem 3rem;
    background-color:#ffe888;
    border: none;
    border-radius: 75px;
    margin-bottom: 1rem;
    color: #633fff;
}

.description {
    font-size: 1.25rem;
    margin-top: 5rem;
    font-style: oblique;
    font-family:Verdana, Geneva, Tahoma, sans-serif;
    color: #ffe888;
}

.el-header{
    background-color: #ffe888;
    color: #633fff;
    text-align: center;
    font-size: 2rem;
    line-height: 60px;
}

.el-container{
    justify-content: center;
    align-content: center;
    height: 100vh;
}


.el-main {
    color: #333;
    text-align: center;
    line-height: 100px;

}

.el-footer {
    background-color: #ffe888;
    color: #633fff;
    text-align: center;
    line-height: 60px;
}

.el-row {
    display: flex;
    justify-content: center;
    align-items: center;

}

.bg-purple {
    background: #ffe888 ;
    margin-top: 50px;
}
.grid-content {
    border-radius: 10px;
    height: 70vh;
    width: 70vh;
    margin-left: 5vh;
}

.btn{
    font-size: 1.5rem;
    padding: 1rem 1rem;
    background-color: #ffe888;
    border: none;
    border-radius: 10px;
    color: #633fff;
}
.image-preview {
    display: inline-block;
    position: relative;
    width: 100%;
    height: 100%;
    border: 2px dashed #ccc;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    margin-bottom: 10px; /* 图片预览和按钮之间添加间距 */
}

.image-preview img {
    width: auto;
    height: 100%;
    max-width: 100%;
    max-height: 100%;
    object-fit: contain; /* 保持图片比例，完整显示在容器内 */
}


</style>
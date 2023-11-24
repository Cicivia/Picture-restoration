<template>

  <!--网站首页-->
  <div class="container">
    <button class="title">Image Restoration</button>
    <p class="description" style="white-space: pre-wrap">
      {{'Upload an image contaminated by noise and return the repair result of the image.'}}
    </p>
  </div>

  <div class="common-layout">
    <el-container>
      
        <el-header>图像修复</el-header>

        <el-container class="container">
          <el-main class="detail">

            <el-row :gutter="40">
              <el-col :span="11">
                <div class="block">
                  <div class="grid-content bg-purple">

                  </div>
                  <button class="btn">上传需修复的图片</button>
                </div>
                
              </el-col>

              <el-col :span="11">
                <div class="block">
                  <div class="grid-content bg-purple">

                  </div>
                  <button class="btn">下载修复后的图片</button>
                </div>
              </el-col>
            </el-row>

          </el-main>
        </el-container>
        

 
        <el-footer>第十二小组</el-footer>

    </el-container>
  </div>




</template>

<style>
  /**网站首页 */
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
    background: #ffeeaa;
    margin-top: 50px;
  }
  .grid-content {
    border-radius: 10px;
    height: 50vh;
  }

  .btn{
    font-size: 1.5rem;
    padding: 1rem 1rem;
    background-color:#ffe888;
    border: none;
    border-radius: 10px;
    color: #633fff;
  }

  
</style>

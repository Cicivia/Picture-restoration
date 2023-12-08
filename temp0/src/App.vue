<template>

<el-carousel :autoplay="false" height="100vh">
    <el-carousel-item>
      <!--网站首页-->
      <div class="container">
        <button class="title">Image Restoration</button>
        <p class="description" style="white-space: pre-wrap">
          {{'Upload an image contaminated by noise and return the repair result of the image.'}}
        </p>
      </div>
    </el-carousel-item>

    <el-carousel-item>
      <div class="common-layout">
        <el-container>
          <el-header>网站介绍</el-header>
          <el-container class="container">
            <el-main class="detail">
              <el-collapse v-model="activeName" accordion>
                <el-collapse-item class="tip1" title="功能 Features" name="1">
                  <div class="content">图像修复：网站提供图像去噪服务，用户可以上传损坏的图像，网站将通过深度学习和图像处理算法对图像进行分析和修复，最终生成清晰、高质量的修复图像。</div>
                  <div class="content">多平台支持：网站支持在各种设备上使用，包括PC端、移动端等，方便用户随时随地进行图像修复操作。</div>
                </el-collapse-item>
                <el-collapse-item class="tip2" title="一致性 Consistency" name="2">
                  <div class="content">与现实生活一致：与现实生活的流程、逻辑保持一致，遵循用户习惯的语言和概念；</div>
                  <div class="content">在界面中一致：所有的元素和结构需保持一致，比如：设计样式、图标和文本、元素的位置等。</div>
                </el-collapse-item>
                <el-collapse-item class="tip1" title="反馈 Feedback" name="3">
                  <div class="content">控制反馈：通过界面样式和交互动效让用户可以清晰的感知自己的操作；</div>
                  <div class="content">页面反馈：操作后，通过页面元素的变化清晰地展现当前状态。</div>
                  <div class="content">用户友好的操作界面：网站提供简洁直观的操作界面，用户可以轻松上传图像进行修复，无需专业的图像处理知识即可完成修复操作。</div>
                </el-collapse-item>
                <el-collapse-item class="tip2" title="效率 Efficiency" name="4">
                  <div class="content">简化流程：设计简洁直观的操作流程；</div>
                  <div class="content">清晰明确：语言表达清晰且表意明确，让用户快速理解进而作出决策；</div>
                  <div class="content">帮助用户识别：界面简单直白，让用户快速识别而非回忆，减少用户记忆负担。</div>
                  <div class="content">高效算法：网站采用先进的深度学习和图像处理算法，能够快速准确地对损坏的图像进行修复，保留原始图像的细节和质量。</div>
                </el-collapse-item>
                <el-collapse-item class="tip1" title="可控 Controllability" name="5">
                  <div class="content">用户决策：根据场景可给予用户操作建议或安全提示，但不能代替用户进行决策；</div>
                  <div class="content">结果可控：用户可以自由的进行操作，包括撤销、回退和终止当前操作等。</div>
                  <div class="content">数据安全保障：网站承诺对用户上传的图像数据进行严格保密，确保用户数据安全。</div>
                </el-collapse-item>
              </el-collapse>
            </el-main>
          </el-container>
        </el-container>
      </div>
    </el-carousel-item>

    <el-carousel-item>
      <div class="common-layout">
        <el-container>
            <el-header>在线使用</el-header>
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
        </el-container>
      </div>
    </el-carousel-item>
  </el-carousel>
</template>


<script>
  export default {
    data() {
      return {
        activeName: '1'
      };
    }
  }
</script>


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


    /* 修改折叠面板的标题样式 */
  .el-collapse-item__header {
    background-color: #fff1bb!important; /* 修改背景颜色 */
    color: #333!important; /* 修改文字颜色 */
    font-size: 20px!important; /* 修改字体大小 */
    padding: 10px; /* 修改内边距 */
    border-radius: 10px;
  }

  /* 修改折叠面板的内容样式 */
  .el-collapse-item__wrap {
    background-color: #fff1bb!important; /* 修改背景颜色 */
    padding: 10px; /* 修改内边距 */
    border-radius: 10px;

  }

  .content {
  font-size: 16px; /* 设置内容的字体大小为16像素 */
  color: #333; /* 设置内容的字体颜色 */
  font-family: Arial, sans-serif; /* 设置内容的字体族 */
}

  

</style>



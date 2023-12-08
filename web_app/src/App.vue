<template>
    <display/>
    <Title />

</template>

<script>
import Title from './components/Title.vue';
import Display from './components/display.vue'
export default {
    name: 'App',
    components: {
        Title,
        Display,
    },
};
</script>

<style>
#app {
    display: flow;
    height: 100%;
    /*background-image: url('./assets/img.png');*/
    background-size: cover;
    background-position: center;
    opacity: 0.9;
}

.information{
    background-color: #42b983;
    color: white;
    text-align: center;

}
.bottem{
    height: 100px;
}

</style>
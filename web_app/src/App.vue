<template>
    <div class="app">
        <div class="information">
            <h3>title</h3>
        </div>
        <Title msg="start"/>
        <div class="bottem">

        </div>
    </div>
</template>

<script>
import Title from './components/Title.vue';

export default {
    name: 'App',
    components: {
        Title,
    },
};
</script>

<style>
.app {
    display: flow;
    height: 100%;
    background-image: url('./assets/img.png');
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
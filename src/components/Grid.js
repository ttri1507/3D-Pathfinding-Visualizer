import React, { useState, useRef, useEffect } from 'react'
import { MeshLambertMaterial } from 'three';
import Tiles from './Tiles';
import { Component } from 'react'
import { Canvas } from "react-three-fiber";

 let arr = new Array(900);

function initializeGrid(){
  let tempGrid = [];
  for(let i = 0; i < 30; i++){
      let currentRow = [];
      for(let j = 0; j < 30; j++){
          let node = createNode(i, j);
          currentRow.push(node);
      }
      tempGrid.push(currentRow);
  }

  for(let i = 0; i < 30; i++){
      arr[i] = 0;
  }
 return tempGrid;

}

function createNode(row, col){

  let node = {
      id: row * col + col,
      row: row,
      col: col,
      distance: Infinity,
      totalDistance: Infinity,
      heuristicDistance: null,
      direction: null,
      weight: 0,
      previousNode: null,
  };

  return node;
}

function Grid(props) {
    const mesh = useRef(null);
    {/*Declare and initialize state variables/ */}
    const [grid, setGrid] = useState([]);

    useEffect(() => {
      // code to run on component mount
      setGrid(initializeGrid());
    }, []);

    const loader = useMemo(() => new THREE.TextureLoader().load(img,
      function(texture){
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.repeat.x = grid.length;
          texture.repeat.y = grid[0].length;
      }), [img]);


    return (
      <mesh ref = {mesh} position = {[0,0,0]}>
        <gridHelper args = {[props.gridDimensions, props.gridDimensions, "hotpink", "hotpink"] }/>
        <meshLambertMaterial attach = "material" >
          <primitive attach = "map" object = {loader} side = {THREE.FrontSide} vertexColors = {THREE.FaceColors}/>
        </meshLambertMaterial>
      <axesHelper />
      </mesh>
  )
}


export default Grid




    
'use client'

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface Conversation {
  id: string;
  content: string;
  children: Conversation[];
  position: [number, number, number];
  color: string;
  branchId: string;
}

interface MementoFractals3DProps {
  conversationHistory: Conversation[];
}

const MementoFractals3D: React.FC<MementoFractals3DProps> = ({ conversationHistory }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<Conversation | null>(null);
  const [selectedNode, setSelectedNode] = useState<Conversation | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const textLabelsRef = useRef<THREE.Sprite[]>([]);

  useEffect(() => {
    if (!mountRef.current) return;

    // Three.js setup
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    sceneRef.current = scene;
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    cameraRef.current = camera;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;
    camera.position.set(0, 0, 50);
    controls.update();

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    // Position nodes in 3D space
    const positionNodes = (conversations: Conversation[], startAngle = 0, radius = 10, y = 0) => {
      const angleStep = (2 * Math.PI) / conversations.length;
      conversations.forEach((conversation, index) => {
        const angle = startAngle + index * angleStep;
        conversation.position = [
          radius * Math.cos(angle),
          y,
          radius * Math.sin(angle)
        ];
        if (conversation.children.length > 0) {
          positionNodes(conversation.children, angle, radius * 0.8, y - 5);
        }
      });
    };

    // Create conversation nodes
    const createNode = (conversation: Conversation, parent?: THREE.Vector3) => {
      const geometry = new THREE.SphereGeometry(0.5, 32, 32);
      const material = new THREE.MeshPhongMaterial({ color: conversation.color });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(...conversation.position);
      sphere.userData = conversation;
      scene.add(sphere);

      // Create text sprite
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      canvas.width = 256;
      canvas.height = 256;
      if (context) {
        context.font = 'Bold 20px Arial';
        context.fillStyle = 'rgba(255,255,255,0.95)';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'black';
        wrapText(context, conversation.content, 10, 30, 236, 25);
      }
      const texture = new THREE.CanvasTexture(canvas);
      const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
      const sprite = new THREE.Sprite(spriteMaterial);
      sprite.scale.set(2, 2, 1);
      sprite.position.set(sphere.position.x, sphere.position.y + 1, sphere.position.z);
      scene.add(sprite);
      textLabelsRef.current.push(sprite);

      // Add line to parent
      if (parent) {
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...conversation.position),
          parent
        ]);
        const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff }); // White color
        const line = new THREE.Line(lineGeometry, lineMaterial);
        scene.add(line);
      }

      conversation.children.forEach((child) => {
        createNode(child, new THREE.Vector3(...conversation.position));
      });
    };

    positionNodes(conversationHistory);
    conversationHistory.forEach(conversation => createNode(conversation));

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const onMouseMove = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, true);

      if (intersects.length > 0) {
        const object = intersects[0].object;
        if (object.userData && object.userData.content) {
          setHoveredNode(object.userData as Conversation);
        } else {
          setHoveredNode(null);
        }
      } else {
        setHoveredNode(null);
      }
    };

    const onClick = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, true);

      if (intersects.length > 0) {
        const object = intersects[0].object;
        if (object.userData && object.userData.content) {
          handleNodeClick(object.userData as Conversation);
        }
      }
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('click', onClick);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      updateTextVisibility();
      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      const newWidth = mountRef.current?.clientWidth || window.innerWidth;
      const newHeight = mountRef.current?.clientHeight || window.innerHeight;
      camera.aspect = newWidth / newHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, newHeight);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('click', onClick);
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [conversationHistory]);

  const handleNodeClick = (conversation: Conversation) => {
    if (cameraRef.current && controlsRef.current) {
      const targetPosition = new THREE.Vector3(...conversation.position);
      const duration = 1000; // Duration of the animation in milliseconds
      const startPosition = cameraRef.current.position.clone();
      const startTime = Date.now();

      const animateCamera = () => {
        const now = Date.now();
        const progress = Math.min((now - startTime) / duration, 1);
        const easeProgress = progress * (2 - progress); // Ease out quadratic

        cameraRef.current!.position.lerpVectors(startPosition, targetPosition, easeProgress);
        cameraRef.current!.lookAt(targetPosition);
        controlsRef.current!.target.copy(targetPosition);

        if (progress < 1) {
          requestAnimationFrame(animateCamera);
        } else {
          // When animation is complete, set the selected node
          setSelectedNode(conversation);
        }
      };

      animateCamera();
    }
  };

  const handleZoomOut = () => {
    if (cameraRef.current && controlsRef.current && sceneRef.current) {
      setSelectedNode(null); // Clear the selected node when zooming out
      const box = new THREE.Box3().setFromObject(sceneRef.current);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());

      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = cameraRef.current.fov * (Math.PI / 180);
      let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));

      cameraZ *= 1.5; // Zoom out a bit more to ensure everything is in view

      const duration = 1000; // Duration of the animation in milliseconds
      const startPosition = cameraRef.current.position.clone();
      const targetPosition = center.clone().add(new THREE.Vector3(0, 0, cameraZ));
      const startTime = Date.now();

      const animateCamera = () => {
        const now = Date.now();
        const progress = Math.min((now - startTime) / duration, 1);
        const easeProgress = progress * (2 - progress); // Ease out quadratic

        cameraRef.current!.position.lerpVectors(startPosition, targetPosition, easeProgress);
        controlsRef.current!.target.copy(center);

        if (progress < 1) {
          requestAnimationFrame(animateCamera);
        } else {
          cameraRef.current!.lookAt(center);
          controlsRef.current!.update();
        }
      };

      animateCamera();
    }
  };

  const updateTextVisibility = () => {
    if (cameraRef.current) {
      textLabelsRef.current.forEach((sprite) => {
        const distance = cameraRef.current!.position.distanceTo(sprite.position);
        const opacity = Math.max(0, Math.min(1, 2 - distance / 10));
        (sprite.material as THREE.SpriteMaterial).opacity = opacity;
      });
    }
  };

  const wrapText = (context: CanvasRenderingContext2D, text: string, x: number, y: number, maxWidth: number, lineHeight: number) => {
    const words = text.split(' ');
    let line = '';

    for (let n = 0; n < words.length; n++) {
      const testLine = line + words[n] + ' ';
      const metrics = context.measureText(testLine);
      const testWidth = metrics.width;
      if (testWidth > maxWidth && n > 0) {
        context.fillText(line, x, y);
        line = words[n] + ' ';
        y += lineHeight;
      } else {
        line = testLine;
      }
    }
    context.fillText(line, x, y);
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      {selectedNode && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          backgroundColor: 'white',
          color: 'black',
          padding: '20px',
          overflowY: 'auto',
          zIndex: 1000
        }}>
          <h2>{selectedNode.id}</h2>
          <p>{selectedNode.content}</p>
          <p>Branch: {selectedNode.branchId}</p>
          <button onClick={() => setSelectedNode(null)}>Close</button>
        </div>
      )}
      <button
        onClick={handleZoomOut}
        style={{
          position: 'absolute',
          bottom: '20px',
          right: '20px',
          padding: '10px 20px',
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}
      >
        Zoom Out
      </button>
      {hoveredNode && !selectedNode && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          backgroundColor: 'rgba(0,0,0,0.7)',
          color: 'white',
          padding: '10px',
          borderRadius: '5px',
          maxWidth: '300px'
        }}>
          <h3>{hoveredNode.content}</h3>
          <p>ID: {hoveredNode.id}</p>
          <p>Branch: {hoveredNode.branchId}</p>
        </div>
      )}
    </div>
  );
};

export default MementoFractals3D;
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
}

interface MementoFractals3DProps {
  conversationHistory: Conversation[];
}

const MementoFractals3D: React.FC<MementoFractals3DProps> = ({ conversationHistory }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const [hoveredNode, setHoveredNode] = useState<Conversation | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Three.js setup
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controlsRef.current = controls;
    camera.position.z = 15;

    // Create conversation nodes
    const createNode = (conversation: Conversation, parent?: THREE.Vector3) => {
      const geometry = new THREE.SphereGeometry(0.5, 32, 32);
      const material = new THREE.MeshPhongMaterial({ color: conversation.color });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.set(...conversation.position);
      scene.add(sphere);

      // Add text to sphere
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d');
      if (context) {
        canvas.width = 256;
        canvas.height = 256;
        context.fillStyle = '#ffffff';
        context.font = '24px Arial';
        context.textAlign = 'center';
        context.fillText(conversation.content.substring(0, 20), 128, 128);
      }
      const texture = new THREE.CanvasTexture(canvas);
      const textMaterial = new THREE.MeshBasicMaterial({ map: texture, transparent: true });
      const textGeometry = new THREE.PlaneGeometry(1, 1);
      const textMesh = new THREE.Mesh(textGeometry, textMaterial);
      textMesh.position.set(0, 0, 0.51); // Slightly in front of the sphere
      sphere.add(textMesh);

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

      // Add hover effect
      sphere.userData = conversation;
      sphere.addEventListener('pointerover', () => setHoveredNode(conversation));
      sphere.addEventListener('pointerout', () => setHoveredNode(null));

      conversation.children.forEach((child) => {
        createNode(child, new THREE.Vector3(...conversation.position));
      });
    };

    conversationHistory.forEach(conversation => createNode(conversation));

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const pointLight = new THREE.PointLight(0xffffff, 1);
    pointLight.position.set(10, 10, 10);
    scene.add(pointLight);

    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();

    const onMouseMove = (event: MouseEvent) => {
      mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
      mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const intersects = raycaster.intersectObjects(scene.children, true);

      if (intersects.length > 0) {
        const object = intersects[0].object;
        if (object.userData && object.userData.content) {
          setHoveredNode(object.userData);
        } else {
          setHoveredNode(null);
        }
      } else {
        setHoveredNode(null);
      }
    };

    window.addEventListener('mousemove', onMouseMove);

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
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

    // Keyboard controls
    const handleKeyDown = (event: KeyboardEvent) => {
      const speed = 0.5;
      switch (event.key) {
        case 'ArrowUp':
          camera.position.y += speed;
          break;
        case 'ArrowDown':
          camera.position.y -= speed;
          break;
        case 'ArrowLeft':
          camera.position.x -= speed;
          break;
        case 'ArrowRight':
          camera.position.x += speed;
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('mousemove', onMouseMove);
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [conversationHistory]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100vh' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
      {hoveredNode && (
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
        </div>
      )}
    </div>
  );
};

export default MementoFractals3D;
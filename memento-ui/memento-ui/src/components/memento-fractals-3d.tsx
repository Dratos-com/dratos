'use client'

import React, { useRef, useState, useCallback, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Text, Html } from '@react-three/drei'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'

type ConversationNode = {
  id: string
  content: string
  children: ConversationNode[]
  position: [number, number, number]
  color: string
}

const mockConversation: ConversationNode = {
  id: 'root',
  content: 'Initial conversation',
  position: [0, 0, 0],
  color: '#4CAF50',
  children: [
    {
      id: 'branch1',
      content: 'AI response',
      position: [2, 2, 0],
      color: '#2196F3',
      children: [
        {
          id: 'branch1-1',
          content: 'User follow-up',
          position: [4, 3, 1],
          color: '#FFC107',
          children: []
        },
        {
          id: 'branch1-2',
          content: 'Alternative response',
          position: [4, 1, -1],
          color: '#9C27B0',
          children: []
        }
      ]
    },
    {
      id: 'branch2',
      content: 'User question',
      position: [-2, -2, 0],
      color: '#E91E63',
      children: [
        {
          id: 'branch2-1',
          content: 'AI clarification',
          position: [-4, -3, 1],
          color: '#00BCD4',
          children: []
        }
      ]
    }
  ]
}

const Node: React.FC<{ node: ConversationNode; onClick: (node: ConversationNode) => void }> = ({ node, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.x += 0.01
      meshRef.current.rotation.y += 0.01
    }
  })

  return (
    <group position={node.position}>
      <mesh
        ref={meshRef}
        onClick={() => onClick(node)}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        <sphereGeometry args={[0.5, 32, 32]} />
        <meshStandardMaterial color={hovered ? '#ffffff' : node.color} />
      </mesh>
      <Html distanceFactor={10}>
        <div className="bg-black bg-opacity-50 text-white p-2 rounded">
          {node.content}
        </div>
      </Html>
    </group>
  )
}

const Edge: React.FC<{ start: [number, number, number]; end: [number, number, number] }> = ({ start, end }) => {
  const ref = useRef<THREE.Line>(null)

  useFrame(() => {
    if (ref.current) {
      ref.current.geometry.setFromPoints([new THREE.Vector3(...start), new THREE.Vector3(...end)])
    }
  })

  return (
    <line ref={ref}>
      <bufferGeometry />
      <lineBasicMaterial color="#ffffff" linewidth={2} />
    </line>
  )
}

const ConversationTree: React.FC<{ root: ConversationNode; onNodeClick: (node: ConversationNode) => void }> = ({ root, onNodeClick }) => {
  const renderNode = useCallback((node: ConversationNode) => {
    return (
      <group key={node.id}>
        <Node node={node} onClick={onNodeClick} />
        {node.children.map(child => (
          <group key={child.id}>
            <Edge start={node.position} end={child.position} />
            {renderNode(child)}
          </group>
        ))}
      </group>
    )
  }, [onNodeClick])

  return renderNode(root)
}

const MementoFractals: React.FC<{ conversationHistory: ConversationNode }> = ({ conversationHistory }) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();

    renderer.setSize(window.innerWidth, window.innerHeight);
    mountRef.current.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);

    const createNode = (node: ConversationNode, position: THREE.Vector3) => {
      const geometry = new THREE.SphereGeometry(0.1, 32, 32);
      const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
      const sphere = new THREE.Mesh(geometry, material);
      sphere.position.copy(position);
      scene.add(sphere);

      node.children.forEach((child, index) => {
        const childPosition = new THREE.Vector3(
          position.x + Math.cos(index * Math.PI * 2 / node.children.length),
          position.y + 1,
          position.z + Math.sin(index * Math.PI * 2 / node.children.length)
        );
        createNode(child, childPosition);

        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints([position, childPosition]),
          new THREE.LineBasicMaterial({ color: 0xffffff })
        );
        scene.add(line);
      });
    };

    createNode(conversationHistory, new THREE.Vector3(0, 0, 0));

    camera.position.z = 5;

    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    return () => {
      mountRef.current?.removeChild(renderer.domElement);
    };
  }, [conversationHistory]);

  return <div ref={mountRef} />;
};

export default MementoFractals
'use client'

import React, { useRef, useState, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Text, Html } from '@react-three/drei'
import * as THREE from 'three'

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

const MementoFractals: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<ConversationNode | null>(null)

  const handleNodeClick = (node: ConversationNode) => {
    setSelectedNode(node)
  }

  return (
    <div className="w-full h-screen">
      <Canvas camera={{ position: [0, 0, 20], fov: 75 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} />
        <ConversationTree root={mockConversation} onNodeClick={handleNodeClick} />
        <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
      </Canvas>
      {selectedNode && (
        <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white p-4">
          <h2 className="text-xl font-bold mb-2">Selected Node: {selectedNode.id}</h2>
          <p>{selectedNode.content}</p>
        </div>
      )}
    </div>
  )
}

export default MementoFractals
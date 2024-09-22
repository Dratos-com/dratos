'use client'

import React, { useEffect, useState } from 'react';
import MementoFractals3D from "@/components/memento-fractals-3d";
import { fetchConversationPage } from '@/lib/api';

interface Conversation {
  id: string;
  content: string;
  children: Conversation[];
  position: [number, number, number];
  color: string;
}

export default function MementoFractalsPage() {
  const [conversationHistory, setConversationHistory] = useState<Conversation[]>([]);

  useEffect(() => {
    const loadConversation = async () => {
      try {
        const data = await fetchConversationPage('initial', 1, 20);
        if (data.messages.length === 0) {
          // Create a default message if the conversation is empty
          setConversationHistory([{
            id: 'initial',
            content: 'Hello, how can I help?',
            children: [],
            position: [0, 0, 0],
            color: '#4CAF50'
          }]);
        } else {
          // Transform the fetched data into the Conversation structure
          const transformedData = transformConversationData(data.messages);
          setConversationHistory(transformedData);
        }
      } catch (error) {
        console.error('Error loading conversation:', error);
        // Set a default message in case of error
        setConversationHistory([{
          id: 'error',
          content: 'Error loading conversation. Please try again.',
          children: [],
          position: [0, 0, 0],
          color: '#FF5722'
        }]);
      }
    };

    loadConversation();
  }, []);

  return (
    <div className="w-full h-screen">
      <MementoFractals3D conversationHistory={conversationHistory} />
    </div>
  );
}

function transformConversationData(messages: any[]): Conversation[] {
  // This is a simple transformation. You may need to adjust it based on your actual data structure
  return messages.map((message, index) => ({
    id: message.id,
    content: message.content,
    children: [],
    position: [index * 2, 0, 0], // Simple linear positioning
    color: message.sender === 'user' ? '#2196F3' : '#4CAF50'
  }));
}
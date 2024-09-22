'use client'

import React, { useState, useEffect } from 'react';
import { ulid } from 'ulid';
import { fetchBranches, fetchConversationPage, createBranch } from '@/lib/api';

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { GitBranch, Send, Edit2, Check, X } from "lucide-react"
import { sendMessage } from "@/lib/api"

type Message = {
  id: string
  content: string
  sender: 'user' | 'ai'
  timestamp: string
}

type Conversation = {
  id: string
  messages: Message[]
}

type Branch = {
  name: string;
  commit_id: string;
  last_modified: string;
};

type SideBySideChatViewerProps = {
  onCreateBranch: (conversationId: string, newBranchId: string) => Promise<void>
  onMergeBranches: (sourceBranch: string, targetBranch: string) => Promise<void>
  onSendMessage: (conversationId: string, content: string, sender: 'user' | 'ai') => Promise<void>
  onEditMessage: (conversationId: string, messageId: string, newContent: string) => Promise<void>
}

export function SideBySideChatViewer({
  onCreateBranch,
  onMergeBranches,
  onSendMessage,
  onEditMessage,
}: SideBySideChatViewerProps) {
  const [conversations, setConversations] = useState<{ [key: string]: Conversation }>({});
  const [activeConversations, setActiveConversations] = useState<string[]>([]);
  const [branches, setBranches] = useState<Branch[]>([]);
  const [newMessage, setNewMessage] = useState('')
  const [editingMessage, setEditingMessage] = useState<string | null>(null)
  const [editContent, setEditContent] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const loadAllBranchesAndConversations = async () => {
      try {
        const fetchedBranches = await fetchBranches('initial');
        setBranches(fetchedBranches);

        const conversationsData: { [key: string]: Conversation } = {};
        for (const branch of fetchedBranches) {
          const conversationData = await fetchConversationPage(branch.name, 1, 20);
          conversationsData[branch.name] = {
            id: branch.name,
            messages: conversationData.messages,
          };
        }
        setConversations(conversationsData);

        // Set active conversations to the first two branches or less if there are fewer branches
        setActiveConversations(fetchedBranches.slice(0, 2).map(branch => branch.name));
      } catch (error) {
        console.error('Error loading branches and conversations:', error);
        setError('Failed to load branches and conversations');
      }
    };

    loadAllBranchesAndConversations();
  }, []);

  const handleSendMessage = async (conversationId: string) => {
    try {
      await onSendMessage(conversationId, newMessage, 'user')
      setNewMessage('')
      // Refresh the conversation after sending a message
      const updatedConversation = await fetchConversationPage(conversationId, 1, 20);
      setConversations(prev => ({
        ...prev,
        [conversationId]: {
          ...prev[conversationId],
          messages: updatedConversation.messages,
        },
      }));
    } catch (error) {
      console.error('Error sending message:', error)
      setError('Failed to send message')
    }
  };

  const handleEditMessage = async (conversationId: string, messageId: string, newContent: string) => {
    try {
      await onEditMessage(conversationId, messageId, newContent)
      setEditingMessage(null)
      // Refresh the conversation after editing a message
      const updatedConversation = await fetchConversationPage(conversationId, 1, 20);
      setConversations(prev => ({
        ...prev,
        [conversationId]: {
          ...prev[conversationId],
          messages: updatedConversation.messages,
        },
      }));
    } catch (error) {
      console.error('Error editing message:', error)
      setError('Failed to edit message')
    }
  };

  const handleCreateBranch = async (conversationId: string) => {
    try {
      const newBranchId = ulid();
      await onCreateBranch(conversationId, newBranchId);
      const updatedBranches = await fetchBranches('initial');
      setBranches(updatedBranches);
      
      // Fetch the new branch's conversation
      const newConversation = await fetchConversationPage(newBranchId, 1, 20);
      setConversations(prev => ({
        ...prev,
        [newBranchId]: {
          id: newBranchId,
          messages: newConversation.messages,
        },
      }));

      // Add the new branch to active conversations if there's room
      if (activeConversations.length < 2) {
        setActiveConversations(prev => [...prev, newBranchId]);
      }
    } catch (error) {
      console.error('Error creating branch:', error);
      setError('Failed to create branch');
    }
  };

  const renderConversation = (conversationId: string) => {
    const conversation = conversations[conversationId]
    if (!conversation) return null

    return (
      <Card className="flex flex-col h-full">
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            Conversation {conversation.id}
            <Button variant="outline" size="sm" onClick={() => handleCreateBranch(conversation.id)}>
              <GitBranch className="w-4 h-4 mr-2" />
              Branch
            </Button>
          </CardTitle>
        </CardHeader>
        <CardContent className="flex-grow overflow-hidden">
          <ScrollArea className="h-[calc(100vh-300px)]">
            {conversation.messages.map(message => (
              <div key={message.id} className={`mb-4 ${message.sender === 'user' ? 'text-right' : 'text-left'}`}>
                <div className={`inline-block p-2 rounded-lg ${message.sender === 'user' ? 'bg-blue-100' : 'bg-gray-100'}`}>
                  {editingMessage === message.id ? (
                    <div className="flex items-center">
                      <Textarea
                        value={editContent}
                        onChange={(e) => setEditContent(e.target.value)}
                        className="min-w-[200px]"
                      />
                      <Button variant="ghost" size="sm" onClick={() => handleEditMessage(conversation.id, message.id, editContent)}>
                        <Check className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => setEditingMessage(null)}>
                        <X className="w-4 h-4" />
                      </Button>
                    </div>
                  ) : (
                    <>
                      <p>{message.content}</p>
                      <Button variant="ghost" size="sm" onClick={() => {
                        setEditingMessage(message.id);
                        setEditContent(message.content);
                      }}>
                        <Edit2 className="w-4 h-4" />
                      </Button>
                    </>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-1">{new Date(message.timestamp).toLocaleString()}</p>
              </div>
            ))}
          </ScrollArea>
        </CardContent>
        <CardFooter>
          <div className="flex w-full items-center space-x-2">
            <Input
              type="text"
              placeholder="Type your message..."
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(conversation.id)}
            />
            <Button type="submit" size="icon" onClick={() => handleSendMessage(conversation.id)}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </CardFooter>
      </Card>
    )
  }

  return (
    <div className="container mx-auto p-4">
      <Tabs defaultValue="side-by-side" className="w-full">
        <TabsList>
          <TabsTrigger value="side-by-side">Side by Side</TabsTrigger>
          <TabsTrigger value="single">Single View</TabsTrigger>
        </TabsList>
        <TabsContent value="side-by-side" className="w-full">
          <div className="grid grid-cols-2 gap-4">
            {activeConversations.map(conversationId => renderConversation(conversationId))}
          </div>
        </TabsContent>
        <TabsContent value="single" className="w-full">
          <div className="grid grid-cols-1 gap-4">
            {renderConversation(activeConversations[0])}
          </div>
        </TabsContent>
      </Tabs>
      {error && <div className="text-red-500 mt-4">{error}</div>}
      <div>
        <h3>Branches</h3>
        <ul>
          {branches.map((branch) => (
            <li key={branch.name}>
              {branch.name} - Last modified: {new Date(branch.last_modified).toLocaleString()}
              <button onClick={() => handleSwitchBranch(branch.name)}>Switch</button>
            </li>
          ))}
        </ul>
        <button onClick={() => handleCreateBranch('initial')}>Create New Branch</button>
      </div>
    </div>
  )
}
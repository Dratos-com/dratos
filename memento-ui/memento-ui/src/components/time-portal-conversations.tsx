'use client'

import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion'
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { GitCommit, ArrowLeft, ArrowRight, Clock, GitBranch, GitFork } from "lucide-react"
import { createBranch, mergeBranches } from '@/lib/api';
const API_BASE_URL = 'http://localhost:8998/api/v1'; // Adjust this to your actual API URL

type Commit = {
  id: string
  message: string
  author: string
  timestamp: string
  changes: {
    added: string[]
    removed: string[]
  }
}

type Branch = {
  id: string
  name: string
  popularity: number
}

type ForkEntry = {
  id: string
  title: string
  author: string
  popularity: number
  comments: number
}

type Conversation = {
  id: string
  title: string
  commits: Commit[]
  branches: Branch[]
  forks: ForkEntry[]
}

interface TimePortalConversationsProps {
  conversationId: string;
  conversation: {
    id: string;
    commits: Commit[];
    branches: Branch[];
    forks: ForkEntry[];
  };
  currentCommitIndex: number;
  onTimeTravel: (direction: 'back' | 'forward') => void;
  onRefresh: () => void;
}

export const TimePortalConversations: React.FC<TimePortalConversationsProps> = ({ conversationId, conversation, currentCommitIndex, onTimeTravel, onRefresh }) => {
  const [conversationHistory, setConversationHistory] = useState([]);
  const [branches, setBranches] = useState([]);
  const [currentBranch, setCurrentBranch] = useState('main');
  const [newBranchName, setNewBranchName] = useState('');
  const [mergeSource, setMergeSource] = useState('');
  const [mergeTarget, setMergeTarget] = useState('');

  const fetchConversationHistory = async (conversationId: string) => {
    const response = await axios.get(`${API_BASE_URL}/conversation/${conversationId}/history`);
    setConversationHistory(response.data);
  };

  const fetchBranches = async (conversationId: string) => {
    const response = await axios.get(`${API_BASE_URL}/conversation/${conversationId}/branches`);
    setBranches(response.data);
  };

  const handleCreateBranch = async () => {
    try {
      await createBranch(conversationId, newBranchName, conversation.commits[currentCommitIndex].id);
      fetchBranches(conversationId);
      onRefresh(); // Refresh the conversation data after creating a branch
    } catch (error) {
      console.error('Error creating branch:', error);
      // Implement user feedback for error (e.g., using a toast notification)
    }
  };

  const handleMergeBranches = async () => {
    try {
      await mergeBranches(conversationId, mergeSource, mergeTarget);
      fetchBranches(conversationId);
      onRefresh(); // Refresh the conversation data after merging branches
    } catch (error) {
      console.error('Error merging branches:', error);
      // Implement user feedback for error (e.g., using a toast notification)
    }
  };

  const switchBranch = async (conversationId: string, branchName: string) => {
    const response = await axios.post(`http://localhost:8998/conversation/${conversationId}/switch-branch`, { branch_name: branchName });
    setConversationHistory(response.data);
    setCurrentBranch(branchName);
  };

  const handleTimeTravel = (direction: 'back' | 'forward') => {
    onTimeTravel(direction);
  };

  useEffect(() => {
    fetchConversationHistory(conversationId);
    fetchBranches(conversationId);
  }, [conversationId]); // Add conversationId to the dependency array

  if (!conversation?.commits) {
    return null; // or some loading/error state
  }

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6 text-center">Time Portal Conversations</h1>
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="col-span-2">
          <div className="flex justify-center items-center mb-4 space-x-4">
            <Button
              variant="outline"
              size="icon"
              onClick={() => handleTimeTravel('back')}
              disabled={currentCommitIndex === 0}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <Card className="w-64">
              <CardHeader className="text-center">
                <CardTitle>Time Portal</CardTitle>
                <CardDescription>
                  Commit {currentCommitIndex + 1} of {conversation.commits.length}
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <Clock className="h-16 w-16 mx-auto mb-2" />
                <p className="font-semibold">{conversation.commits[currentCommitIndex].timestamp}</p>
              </CardContent>
            </Card>
            <Button
              variant="outline"
              size="icon"
              onClick={() => handleTimeTravel('forward')}
              disabled={currentCommitIndex === conversation.commits.length - 1}
            >
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
          <Card className="mb-4">
            <CardHeader>
              <CardTitle>Current Commit</CardTitle>
              <CardDescription>{conversation.commits[currentCommitIndex].message}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2 mb-4">
                <Avatar>
                  <AvatarImage src={`https://api.dicebear.com/6.x/initials/svg?seed=${conversation.commits[currentCommitIndex].author}`} />
                  <AvatarFallback>{conversation.commits[currentCommitIndex].author[0]}</AvatarFallback>
                </Avatar>
                <div>
                  <p className="font-semibold">{conversation.commits[currentCommitIndex].author}</p>
                  <p className="text-sm text-muted-foreground">{conversation.commits[currentCommitIndex].timestamp}</p>
                </div>
              </div>
              <div className="space-y-2">
                {conversation.commits[currentCommitIndex].changes.added.map((change: string, index: number) => (
                  <motion.div
                    key={`added-${index}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                    className="p-2 bg-green-100 dark:bg-green-900 rounded"
                  >
                    <span className="text-green-800 dark:text-green-200">+ {change}</span>
                  </motion.div>
                ))}
                {conversation.commits[currentCommitIndex].changes.removed.map((change: string, index: number) => (
                  <motion.div
                    key={`removed-${index}`}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5 }}
                    className="p-2 bg-red-100 dark:bg-red-900 rounded"
                  >
                    <span className="text-red-800 dark:text-red-200">- {change}</span>
                  </motion.div>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Popular Branches</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {conversation.branches.map((branch) => (
                  <Button key={branch.id} variant="outline" className="flex items-center space-x-2">
                    <GitBranch className="h-4 w-4" />
                    <span>{branch.name}</span>
                    <span className="text-xs bg-primary text-primary-foreground rounded-full px-2 py-1">
                      {branch.popularity}
                    </span>
                  </Button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Conversation Branching</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[400px] flex items-center justify-center bg-accent rounded-md">
              <p className="text-center text-muted-foreground">Minimap visualization coming soon...</p>
            </div>
          </CardContent>
        </Card>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Conversation Details</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Commit</TableHead>
                <TableHead>Author</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Message</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {conversation.commits.map((commit, index) => (
                <TableRow key={commit.id} className={index === currentCommitIndex ? "bg-accent" : ""}>
                  <TableCell>{index + 1}</TableCell>
                  <TableCell>{commit.author}</TableCell>
                  <TableCell>{commit.timestamp}</TableCell>
                  <TableCell>{commit.message}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <Card className="mt-4">
        <CardHeader>
          <CardTitle>Fork Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Title</TableHead>
                <TableHead>Author</TableHead>
                <TableHead>Popularity</TableHead>
                <TableHead>Comments</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {conversation.forks.map((fork) => (
                <TableRow key={fork.id}>
                  <TableCell>{fork.title}</TableCell>
                  <TableCell>{fork.author}</TableCell>
                  <TableCell>{fork.popularity}</TableCell>
                  <TableCell>{fork.comments}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      <div className="mt-4">
        <input
          type="text"
          value={newBranchName}
          onChange={(e) => setNewBranchName(e.target.value)}
          placeholder="New branch name"
          className="border p-2 mr-2"
        />
        <Button onClick={handleCreateBranch}>Create Branch</Button>
      </div>
      <div className="mt-4">
        <input
          type="text"
          value={mergeSource}
          onChange={(e) => setMergeSource(e.target.value)}
          placeholder="Source branch"
          className="border p-2 mr-2"
        />
        <input
          type="text"
          value={mergeTarget}
          onChange={(e) => setMergeTarget(e.target.value)}
          placeholder="Target branch"
          className="border p-2 mr-2"
        />
        <Button onClick={handleMergeBranches}>Merge Branches</Button>
      </div>
    </div>
  )
}
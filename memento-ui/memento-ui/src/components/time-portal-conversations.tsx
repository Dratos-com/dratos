'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { GitCommit, ArrowLeft, ArrowRight, Clock, GitBranch, GitFork } from "lucide-react"

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

const mockConversation: Conversation = {
  id: '1',
  title: 'The Impact of AI on Job Markets',
  commits: [
    {
      id: 'c1',
      message: 'Initial conversation start',
      author: 'Alice',
      timestamp: '2023-06-01 10:00',
      changes: {
        added: ['AI will create new job opportunities in tech sectors.'],
        removed: []
      }
    },
    {
      id: 'c2',
      message: 'Added counterpoint',
      author: 'Bob',
      timestamp: '2023-06-01 11:30',
      changes: {
        added: ['However, AI might also lead to job displacement in certain industries.'],
        removed: []
      }
    },
    {
      id: 'c3',
      message: 'Expanded on AI job creation',
      author: 'Charlie',
      timestamp: '2023-06-01 14:15',
      changes: {
        added: ['AI will likely create jobs in data analysis, machine learning engineering, and AI ethics.'],
        removed: []
      }
    },
    {
      id: 'c4',
      message: 'Added statistics',
      author: 'David',
      timestamp: '2023-06-02 09:45',
      changes: {
        added: ['According to a recent study, AI could automate up to 30% of work globally by 2030.'],
        removed: []
      }
    },
    {
      id: 'c5',
      message: 'Discussed reskilling',
      author: 'Eve',
      timestamp: '2023-06-02 13:20',
      changes: {
        added: ['To adapt to AI-driven changes, workforce reskilling and lifelong learning will be crucial.'],
        removed: ['AI will create new job opportunities in tech sectors.']
      }
    }
  ],
  branches: [
    { id: 'b1', name: 'AI in Healthcare', popularity: 95 },
    { id: 'b2', name: 'AI Ethics', popularity: 88 },
    { id: 'b3', name: 'AI and Education', popularity: 82 },
  ],
  forks: [
    { id: 'f1', title: 'AI\'s Impact on Creative Industries', author: 'Frank', popularity: 76, comments: 42 },
    { id: 'f2', title: 'AI and the Future of Remote Work', author: 'Grace', popularity: 68, comments: 35 },
    { id: 'f3', title: 'AI in Developing Economies', author: 'Henry', popularity: 72, comments: 39 },
  ]
}

export function TimePortalConversations() {
  const [currentCommitIndex, setCurrentCommitIndex] = useState(mockConversation.commits.length - 1)
  const currentCommit = mockConversation.commits[currentCommitIndex]

  const handleTimeTravel = (direction: 'back' | 'forward') => {
    if (direction === 'back' && currentCommitIndex > 0) {
      setCurrentCommitIndex(currentCommitIndex - 1)
    } else if (direction === 'forward' && currentCommitIndex < mockConversation.commits.length - 1) {
      setCurrentCommitIndex(currentCommitIndex + 1)
    }
  }

  return (
    <div className="container mx-auto p-4 max-w-6xl">
      <h1 className="text-3xl font-bold mb-6 text-center">{mockConversation.title}</h1>
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
                  Commit {currentCommitIndex + 1} of {mockConversation.commits.length}
                </CardDescription>
              </CardHeader>
              <CardContent className="text-center">
                <Clock className="h-16 w-16 mx-auto mb-2" />
                <p className="font-semibold">{currentCommit.timestamp}</p>
              </CardContent>
            </Card>
            <Button
              variant="outline"
              size="icon"
              onClick={() => handleTimeTravel('forward')}
              disabled={currentCommitIndex === mockConversation.commits.length - 1}
            >
              <ArrowRight className="h-4 w-4" />
            </Button>
          </div>
          <Card className="mb-4">
            <CardHeader>
              <CardTitle>Current Commit</CardTitle>
              <CardDescription>{currentCommit.message}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center space-x-2 mb-4">
                <Avatar>
                  <AvatarImage src={`https://api.dicebear.com/6.x/initials/svg?seed=${currentCommit.author}`} />
                  <AvatarFallback>{currentCommit.author[0]}</AvatarFallback>
                </Avatar>
                <div>
                  <p className="font-semibold">{currentCommit.author}</p>
                  <p className="text-sm text-muted-foreground">{currentCommit.timestamp}</p>
                </div>
              </div>
              <div className="space-y-2">
                {currentCommit.changes.added.map((change, index) => (
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
                {currentCommit.changes.removed.map((change, index) => (
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
                {mockConversation.branches.map((branch) => (
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
              {mockConversation.commits.map((commit, index) => (
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
              {mockConversation.forks.map((fork) => (
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
    </div>
  )
}
'use client'

import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { GitCommit, ArrowLeft, ArrowRight, GitBranch, GitFork, Search, HelpCircle } from "lucide-react"

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
  description: string
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
    { id: 'b1', name: 'AI in Healthcare', popularity: 95, description: 'Exploring the impact of AI on healthcare jobs and patient care.' },
    { id: 'b2', name: 'AI Ethics', popularity: 88, description: 'Discussing ethical considerations in AI development and deployment.' },
    { id: 'b3', name: 'AI and Education', popularity: 82, description: 'Analyzing how AI is transforming educational jobs and learning methods.' },
  ],
  forks: [
    { id: 'f1', title: 'AI: Impact on Creative Industries', author: 'Frank', popularity: 76, comments: 42 },
    { id: 'f2', title: 'AI and the Future of Remote Work', author: 'Grace', popularity: 68, comments: 35 },
    { id: 'f3', title: 'AI in Developing Economies', author: 'Henry', popularity: 72, comments: 39 },
  ]
}

export function EnhancedTimePortalConversations() {
  const [currentCommitIndex, setCurrentCommitIndex] = useState(mockConversation.commits.length - 1)
  const [isHelpVisible, setIsHelpVisible] = useState(false)
  const timelineRef = useRef<HTMLDivElement>(null)

  const currentCommit = mockConversation.commits[currentCommitIndex]

  useEffect(() => {
    if (timelineRef.current) {
      const activeItem = timelineRef.current.querySelector('.active')
      if (activeItem) {
        activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }
    }
  }, [currentCommitIndex])

  const handleTimeTravel = (direction: 'back' | 'forward') => {
    setCurrentCommitIndex(prevIndex => {
      if (direction === 'back' && prevIndex > 0) {
        return prevIndex - 1
      } else if (direction === 'forward' && prevIndex < mockConversation.commits.length - 1) {
        return prevIndex + 1
      }
      return prevIndex
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      handleTimeTravel('back')
    } else if (e.key === 'ArrowRight') {
      handleTimeTravel('forward')
    }
  }

  return (
    <TooltipProvider>
      <div className="container mx-auto p-4 max-w-6xl" onKeyDown={handleKeyDown} tabIndex={0}>
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold">{mockConversation.title}</h1>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="icon" aria-label="Search">
              <Search className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" onClick={() => setIsHelpVisible(!isHelpVisible)} aria-label="Help">
              <HelpCircle className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <AnimatePresence>
          {isHelpVisible && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-muted p-4 rounded-lg mb-4"
            >
              <h2 className="text-lg font-semibold mb-2">How to use the Time Portal</h2>
              <ul className="list-disc pl-5 space-y-1">
                <li>Use the left and right arrows to navigate through commits</li>
                <li>Click on a commit in the timeline to jump to that point</li>
                <li>Explore branches and forks to see different conversation paths</li>
                <li>Use keyboard arrow keys for quick navigation</li>
              </ul>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="col-span-2 space-y-4">
            <Card>
              <CardHeader className="text-center">
                <CardTitle>Time Portal</CardTitle>
                <CardDescription>
                  Navigating through conversation history
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex justify-center items-center space-x-4">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handleTimeTravel('back')}
                    disabled={currentCommitIndex === 0}
                    aria-label="Previous commit"
                  >
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                  <div className="relative">
                    <motion.div
                      className="w-32 h-32 rounded-full border-4 border-primary flex items-center justify-center overflow-hidden"
                      animate={{ rotate: 360 * (currentCommitIndex / (mockConversation.commits.length - 1)) }}
                      transition={{ type: "spring", stiffness: 100 }}
                    >
                      <div className="text-center">
                        <p className="font-semibold">Commit {currentCommitIndex + 1}</p>
                        <p className="text-sm text-muted-foreground">{currentCommit.timestamp}</p>
                      </div>
                    </motion.div>
                  </div>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => handleTimeTravel('forward')}
                    disabled={currentCommitIndex === mockConversation.commits.length - 1}
                    aria-label="Next commit"
                  >
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
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
                    <Tooltip key={branch.id}>
                      <TooltipTrigger asChild>
                        <Button variant="outline" className="flex items-center space-x-2">
                          <GitBranch className="h-4 w-4" />
                          <span>{branch.name}</span>
                          <span className="text-xs bg-primary text-primary-foreground rounded-full px-2 py-1">
                            {branch.popularity}
                          </span>
                        </Button>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{branch.description}</p>
                      </TooltipContent>
                    </Tooltip>
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
              <div className="h-[400px] relative">
                <svg className="w-full h-full">
                  <line x1="50%" y1="0" x2="50%" y2="100%" stroke="currentColor" strokeWidth="2" />
                  {mockConversation.commits.map((_, index) => (
                    <g key={index} transform={`translate(0, ${(index / (mockConversation.commits.length - 1)) * 100}%)`}>
                      <line x1="25%" y1="0" x2="75%" y2="0" stroke="currentColor" strokeWidth="2" />
                      <circle cx="50%" cy="0" r="4" fill="currentColor" />
                    </g>
                  ))}
                </svg>
                {mockConversation.forks.map((fork, index) => (
                  <Tooltip key={fork.id}>
                    <TooltipTrigger asChild>
                      <Button
                        variant="outline"
                        size="sm"
                        className="absolute left-3/4 transform -translate-x-1/2"
                        style={{ top: `${(index / (mockConversation.forks.length + 1)) * 100}%` }}
                      >
                        <GitFork className="h-4 w-4 mr-2" />
                        {fork.title}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Author: {fork.author}</p>
                      <p>Popularity: {fork.popularity}</p>
                      <p>Comments: {fork.comments}</p>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Commit Timeline</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[100px]">
              <div className="flex space-x-2" ref={timelineRef}>
                {mockConversation.commits.map((commit, index) => (
                  <Tooltip key={commit.id}>
                    <TooltipTrigger asChild>
                      <Button
                        variant={index === currentCommitIndex ? "default" : "outline"}
                        size="sm"
                        className={`flex-shrink-0 ${index === currentCommitIndex ? 'active' : ''}`}
                        onClick={() => setCurrentCommitIndex(index)}
                      >
                        <GitCommit className="h-4 w-4 mr-2" />
                        {index + 1}
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>{commit.message}</p>
                      <p>{commit.author} - {commit.timestamp}</p>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </TooltipProvider>
  )
}
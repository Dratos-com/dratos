'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, GitFork, MessageSquare, Star, TrendingUp, Users } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

type Conversation = {
  id: string
  title: string
  author: string
  content: string
  popularity: number
  forks: number
  comments: number
  timeData: { time: string; popularity: number }[]
}

const mockConversations: Conversation[] = [
  {
    id: '1',
    title: 'The Future of AI in Healthcare',
    author: 'Dr. Smith',
    content: 'AI is revolutionizing healthcare...',
    popularity: 95,
    forks: 12,
    comments: 87,
    timeData: [
      { time: 'Mon', popularity: 20 },
      { time: 'Tue', popularity: 40 },
      { time: 'Wed', popularity: 60 },
      { time: 'Thu', popularity: 80 },
      { time: 'Fri', popularity: 95 },
    ]
  },
  {
    id: '2',
    title: 'Ethical Considerations in AI Development',
    author: 'Prof. Johnson',
    content: 'As AI becomes more prevalent...',
    popularity: 88,
    forks: 8,
    comments: 62,
    timeData: [
      { time: 'Mon', popularity: 30 },
      { time: 'Tue', popularity: 45 },
      { time: 'Wed', popularity: 55 },
      { time: 'Thu', popularity: 70 },
      { time: 'Fri', popularity: 88 },
    ]
  },
  {
    id: '3',
    title: 'AI in Climate Change Mitigation',
    author: 'Dr. Green',
    content: 'AI can play a crucial role...',
    popularity: 92,
    forks: 15,
    comments: 103,
    timeData: [
      { time: 'Mon', popularity: 40 },
      { time: 'Tue', popularity: 55 },
      { time: 'Wed', popularity: 70 },
      { time: 'Thu', popularity: 85 },
      { time: 'Fri', popularity: 92 },
    ]
  },
  {
    id: '4',
    title: 'The Role of AI in Education',
    author: 'Prof. Davis',
    content: 'AI is transforming the way we learn...',
    popularity: 85,
    forks: 7,
    comments: 76,
    timeData: [
      { time: 'Mon', popularity: 25 },
      { time: 'Tue', popularity: 40 },
      { time: 'Wed', popularity: 60 },
      { time: 'Thu', popularity: 75 },
      { time: 'Fri', popularity: 85 },
    ]
  },
]

export function BentoConversationUi() {
  const [selectedConversation, setSelectedConversation] = useState<Conversation | null>(null)

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-3xl font-bold mb-6">Trending Conversations</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <Card className="col-span-1 md:col-span-2 lg:col-span-2 row-span-2">
          <CardHeader>
            <CardTitle>Popular Conversations</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[400px]">
              {mockConversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className="flex items-center space-x-4 mb-4 p-2 hover:bg-accent rounded-md cursor-pointer"
                  onClick={() => setSelectedConversation(conversation)}
                >
                  <Avatar>
                    <AvatarImage src={`https://api.dicebear.com/6.x/initials/svg?seed=${conversation.author}`} />
                    <AvatarFallback>{conversation.author[0]}</AvatarFallback>
                  </Avatar>
                  <div className="flex-1">
                    <h3 className="font-semibold">{conversation.title}</h3>
                    <p className="text-sm text-muted-foreground">{conversation.author}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Star className="w-4 h-4 text-yellow-400" />
                    <span>{conversation.popularity}</span>
                  </div>
                </div>
              ))}
            </ScrollArea>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Trending Topics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span>AI Ethics</span>
                <TrendingUp className="w-4 h-4 text-green-500" />
              </div>
              <div className="flex items-center justify-between">
                <span>Machine Learning</span>
                <TrendingUp className="w-4 h-4 text-green-500" />
              </div>
              <div className="flex items-center justify-between">
                <span>Neural Networks</span>
                <TrendingUp className="w-4 h-4 text-green-500" />
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Activity Overview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span>Total Conversations</span>
                <span className="font-semibold">1,234</span>
              </div>
              <div className="flex items-center justify-between">
                <span>Active Users</span>
                <span className="font-semibold">567</span>
              </div>
              <div className="flex items-center justify-between">
                <span>New Forks Today</span>
                <span className="font-semibold">89</span>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card className="col-span-1 md:col-span-2 lg:col-span-3">
          <CardHeader>
            <CardTitle>Conversation Details</CardTitle>
          </CardHeader>
          <CardContent>
            {selectedConversation ? (
              <div className="space-y-4">
                <h2 className="text-2xl font-bold">{selectedConversation.title}</h2>
                <p className="text-muted-foreground">{selectedConversation.content}</p>
                <div className="flex space-x-4">
                  <div className="flex items-center">
                    <Star className="w-4 h-4 mr-1 text-yellow-400" />
                    <span>{selectedConversation.popularity}</span>
                  </div>
                  <div className="flex items-center">
                    <GitFork className="w-4 h-4 mr-1" />
                    <span>{selectedConversation.forks}</span>
                  </div>
                  <div className="flex items-center">
                    <MessageSquare className="w-4 h-4 mr-1" />
                    <span>{selectedConversation.comments}</span>
                  </div>
                </div>
                <Tabs defaultValue="popularity">
                  <TabsList>
                    <TabsTrigger value="popularity">Popularity Over Time</TabsTrigger>
                    <TabsTrigger value="forks">Fork Activity</TabsTrigger>
                  </TabsList>
                  <TabsContent value="popularity">
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={selectedConversation.timeData}>
                        <XAxis dataKey="time" />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="popularity" stroke="#8884d8" />
                      </LineChart>
                    </ResponsiveContainer>
                  </TabsContent>
                  <TabsContent value="forks">
                    <div className="text-center text-muted-foreground">
                      Fork activity visualization coming soon...
                    </div>
                  </TabsContent>
                </Tabs>
              </div>
            ) : (
              <div className="text-center text-muted-foreground">
                Select a conversation to view details
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
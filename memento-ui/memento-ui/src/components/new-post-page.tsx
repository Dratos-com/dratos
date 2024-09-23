'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { GitBranch, Clock, Tag, AlertTriangle } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"
import { createPost } from '@/lib/api'
import { useRouter } from 'next/navigation'

type TimePoint = {
  id: string
  name: string
  description: string
}

const mockTimePoints: TimePoint[] = [
  { id: 'present', name: 'Present', description: 'Current timeline' },
  { id: 'ancient', name: 'Ancient Times', description: 'Before 500 AD' },
  { id: 'medieval', name: 'Medieval Era', description: '500 AD - 1500 AD' },
  { id: 'renaissance', name: 'Renaissance', description: '14th - 17th century' },
  { id: 'industrial', name: 'Industrial Revolution', description: '18th - 19th century' },
  { id: 'modern', name: 'Modern Era', description: '20th century onwards' },
  { id: 'future', name: 'Future', description: 'Speculative future timelines' },
]

type NewPostPageProps = {
  parentCommit: string
}

export function NewPostPageComponent({ parentCommit }: NewPostPageProps) {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [selectedTimePoint, setSelectedTimePoint] = useState('')
  const [tags, setTags] = useState('')
  const [isParadoxWarningEnabled, setIsParadoxWarningEnabled] = useState(true)
  const [parentPost, setParentPost] = useState('')
  const { addToast } = useToast()
  const router = useRouter()

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    try {
      console.log("Submitting post:", { title, content, author: "CurrentUser", timePoint: selectedTimePoint, tags });  // Add this line
      const newPost = await createPost(
        title,
        content,
        "CurrentUser",
        selectedTimePoint,
        tags
      )
      console.log('New post created:', newPost)
      addToast("Post Created: Your post has been successfully created and added to the timeline.")
      // Reset form fields and redirect
      setTitle('')
      setContent('')
      setSelectedTimePoint('')
      setTags('')
      setParentPost('')
      router.push('/git-versioned-forum')
    } catch (error) {
      console.error('Error creating post:', error)
      if (error.response) {
        console.error('Response data:', error.response.data);
        console.error('Response status:', error.response.status);
        console.error('Response headers:', error.response.headers);
      }
      addToast("Error: There was an error creating your post. Please try again.")
    }
  }

  return (
    <div className="container mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Create a New Post</CardTitle>
          <CardDescription>Share your thoughts across time and space</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <div className="space-y-4">
              <div>
                <Label htmlFor="title">Title</Label>
                <Input
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter your post title"
                  required
                />
              </div>
              <div>
                <Label htmlFor="content">Content</Label>
                <Textarea
                  id="content"
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="Write your post content here..."
                  required
                  className="min-h-[200px]"
                />
              </div>
              <div>
                <Label htmlFor="timePoint">Time Point</Label>
                <Select value={selectedTimePoint} onValueChange={setSelectedTimePoint}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a time point" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockTimePoints.map((timePoint) => (
                      <SelectItem key={timePoint.id} value={timePoint.id}>
                        {timePoint.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label htmlFor="tags">Tags</Label>
                <Input
                  id="tags"
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                  placeholder="Enter tags separated by commas"
                />
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="paradox-warning"
                  checked={isParadoxWarningEnabled}
                  onCheckedChange={setIsParadoxWarningEnabled}
                />
                <Label htmlFor="paradox-warning">Enable paradox warnings</Label>
              </div>
              <div>
                <Label htmlFor="parentPost">Parent Post (for branching)</Label>
                <Input
                  id="parentPost"
                  value={parentPost}
                  onChange={(e) => setParentPost(e.target.value)}
                  placeholder="Enter ID of parent post (if applicable)"
                />
              </div>
            </div>
          </form>
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button variant="outline">Preview</Button>
          <Button type="submit" onClick={handleSubmit}>Create Post</Button>
        </CardFooter>
      </Card>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Time Travel Guidelines</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[200px]">
              <ul className="space-y-2">
                <li className="flex items-center">
                  <Clock className="w-4 h-4 mr-2" />
                  Specify the exact time period for your post
                </li>
                <li className="flex items-center">
                  <GitBranch className="w-4 h-4 mr-2" />
                  Create branches to explore alternative timelines
                </li>
                <li className="flex items-center">
                  <Tag className="w-4 h-4 mr-2" />
                  Use relevant tags to categorize your post
                </li>
                <li className="flex items-center">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  Be aware of potential paradoxes in your discussions
                </li>
              </ul>
            </ScrollArea>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Popular Time Points</CardTitle>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[200px]">
              <ul className="space-y-2">
                {mockTimePoints.map((timePoint) => (
                  <li key={timePoint.id} className="flex items-center justify-between">
                    <span className="font-medium">{timePoint.name}</span>
                    <span className="text-sm text-muted-foreground">{timePoint.description}</span>
                  </li>
                ))}
              </ul>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}